"""Med Pre-Consult (MPC) — AI health-information app by Team Archlve. Educational support, not medical diagnosis or prescribing."""

import base64
import io
import json
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st


# Embedded avatar/logo images are stored as PNG files under assets/ and
# loaded + base64-encoded at runtime, instead of being hardcoded as giant
# base64 string literals in this file.
ASSETS_DIR = Path(__file__).parent / "assets"


def _load_image_b64(filename: str) -> str:
    return base64.b64encode((ASSETS_DIR / filename).read_bytes()).decode("ascii")


NIGHTMARE_AVATAR_B64 = _load_image_b64("nightmare_avatar.png")
ARCHLVE_LOGO_B64 = _load_image_b64("archlve_logo.png")


DB_PATH = Path(__file__).with_name("patient_records.db")
ENV_PATH = Path(__file__).with_name("opai.env")


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE pairs from a local .env-style file into os.environ.

    Existing environment variables are never overwritten, so a value already
    exported in the real shell still takes priority over the file.
    """
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_streamlit_secrets_into_env() -> None:
    """Copy matching keys from st.secrets into os.environ if not already set.

    On Streamlit Community Cloud there is no local opai.env file (it's never pushed
    to GitHub), so secrets are configured instead via the app's Settings -> Secrets
    panel and exposed through st.secrets. This lets the same code work locally
    (via opai.env) and once deployed (via st.secrets) without any other changes.
    """
    try:
        secrets = st.secrets
    except Exception:
        return
    for key in ("GROQ_API_KEY", "GROQ_HEALTH_MODEL", "GROQ_MAX_COMPLETION_TOKENS"):
        if key in os.environ:
            continue
        try:
            value = secrets[key]
        except Exception:
            continue
        if value:
            os.environ[key] = str(value)


load_env_file(ENV_PATH)
load_streamlit_secrets_into_env()


URGENCY_LEVELS = [
    "Emergency now",
    "Urgent same-day assessment",
    "Self-care",
]

REPORT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "urgency": {"type": "string", "enum": URGENCY_LEVELS},
        "report_markdown": {"type": "string"},
    },
    "required": ["urgency", "report_markdown"],
}

URGENCY_ONLY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"urgency": {"type": "string", "enum": URGENCY_LEVELS}},
    "required": ["urgency"],
}

# Headings grouped into a few requests instead of one, so each request's output
# fits under a small per-minute token cap. Groups are generated one per minute.
SECTION_GROUPS = [
    [
        "## What your symptoms may mean",
        "## Possible causes to consider",
        "## Details that change the likelihood",
    ],
    [
        "## Medication and treatment discussion",
        "## Reasonable next steps and tests to discuss",
    ],
    [
        "## Safe self-care while arranging care",
        "## Seek emergency help immediately if",
        "## Questions to take to a clinician",
    ],
]


def initialize_database() -> None:
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS patient_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                patient_name TEXT NOT NULL,
                symptoms TEXT NOT NULL,
                patient_json TEXT NOT NULL,
                urgency TEXT NOT NULL,
                report_markdown TEXT NOT NULL
            )"""
        )


def save_record(patient_name: str, patient: dict, report: dict) -> None:
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """INSERT INTO patient_records
            (created_at, patient_name, symptoms, patient_json, urgency, report_markdown)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(), patient_name, patient["symptoms"],
                json.dumps(patient), report["urgency"], report["report_markdown"],
            ),
        )


def search_records(query: str):
    pattern = f"%{query.strip()}%"
    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        return connection.execute(
            """SELECT * FROM patient_records
            WHERE patient_name LIKE ? OR symptoms LIKE ? OR report_markdown LIKE ?
            ORDER BY created_at DESC""",
            (pattern, pattern, pattern),
        ).fetchall()


def _call_groq_with_backoff(client, model_name, messages, target_tokens, response_format=None, known_limit=None, max_attempts=5):
    """Call Groq chat completions, automatically handling 429 output-token rate limits.

    Two distinct 429 shapes are handled:
    - A hard per-request cap ("Limit N ... output tokens per minute"): the request itself asked
      for more than the account/model allows in one call, so we shrink to fit and retry immediately.
    - A transient cap from tokens already used this minute ("... try again in N.Ns"): we wait out
      the window and retry at the same size.

    Returns (response, known_limit) — known_limit is the discovered per-minute output-token cap
    for this account/model (None if it was never hit), so callers can reuse it for later calls
    instead of re-discovering it from scratch each time.
    """
    from groq import RateLimitError

    tokens = target_tokens if known_limit is None else max(min(target_tokens, known_limit - 50), 300)

    def do_call(t):
        kwargs = dict(model=model_name, messages=messages, max_completion_tokens=t, temperature=0.2)
        if response_format is not None:
            kwargs["response_format"] = response_format
        return client.chat.completions.create(**kwargs)

    attempt = 0
    while True:
        attempt += 1
        try:
            return do_call(tokens), known_limit
        except RateLimitError as error:
            if attempt >= max_attempts:
                raise RuntimeError(f"Groq kept rate-limiting this request: {error}") from error
            message = str(error)
            limit_match = re.search(r"Limit (\d+)", message)
            wait_match = re.search(r"try again in ([\d.]+)s", message, re.IGNORECASE)
            if limit_match and "output tokens per minute" in message.lower():
                known_limit = int(limit_match.group(1))
                tokens = max(known_limit - 50, 300)
                continue
            if wait_match:
                time.sleep(min(float(wait_match.group(1)) + 1, 65))
                continue
            raise RuntimeError(message) from error
        except Exception as error:
            raise RuntimeError(str(error)) from error


def generate_report(patient: dict) -> dict:
    """Return a detailed report written entirely by the AI in a single call; never use a local fallback."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured. Add it, then restart Streamlit.")

    try:
        from groq import Groq
    except ModuleNotFoundError as error:
        raise RuntimeError("Groq package missing. Run: pip install --upgrade groq") from error

    instructions = """
You are a careful clinical-information assistant. Create a thorough, patient-friendly report from the intake.
This is educational triage, not a diagnosis, a prescription, or a substitute for an in-person examination.

Classify this situation into exactly one of the three categories in the supplied list, based solely on
the information the patient submitted below — never on anything not stated:
- "Emergency now": the reported symptoms include or plausibly suggest a red-flag / life-threatening
  presentation (e.g. features consistent with stroke, heart attack, severe difficulty breathing, severe
  bleeding, anaphylaxis, overdose, sepsis, suicidal intent) based on what was described.
- "Urgent same-day assessment": the reported symptoms are concerning enough to need a clinician's
  assessment the same day, but nothing described suggests an immediate life-threatening emergency.
- "Self-care": nothing described suggests urgent or emergency risk, and the symptoms are the kind that
  can reasonably be monitored/self-managed while arranging a routine, non-urgent appointment if needed.

Do not use this category as a substitute for the report: give clinically useful detail in every section.
Return only JSON conforming to the schema.

In report_markdown, use exactly these Markdown H2 headings, in this order:
## What your symptoms may mean
## Possible causes to consider
## Details that change the likelihood
## Medication and treatment discussion
## Reasonable next steps and tests to discuss
## Safe self-care while arranging care
## Seek emergency help immediately if
## Questions to take to a clinician

Requirements:
- Explain 3–6 plausible cause categories when the symptoms allow it. For each, explain why it may fit
  and what details would support or argue against it. Clearly label them as possibilities, never conclusions.
- Give a detailed medication discussion based on the reported medicines, allergies and conditions. Mention
  relevant medication classes or treatment categories only as topics to discuss with a clinician/pharmacist.
  Never prescribe, recommend a dose, provide a dosing schedule, or tell the person to start, stop, or change
  any medicine. Do not present a list of every possible medication as a treatment plan.
- Include practical, low-risk comfort measures when appropriate, plus clear limits on when not to self-manage.
- Include specific red flags relevant to the stated symptoms. If a possible emergency is plausible, place that
  first and choose Emergency now.
- Do not invent examination findings, test results, diagnoses, allergies, pregnancy status, or medical history.
- Do not be terse or merely say 'see a clinic'. Explain what needs assessment and why.
- Complete every required heading, but keep the whole report within 5,000 words so it can finish in one response.
""".strip()

    client = Groq(api_key=api_key)
    model_name = os.getenv("GROQ_HEALTH_MODEL", "qwen/qwen3.8-27b")
    requested_max_tokens = int(os.getenv("GROQ_MAX_COMPLETION_TOKENS", "12000"))
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": json.dumps(patient)},
    ]
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "detailed_health_report", "strict": True, "schema": REPORT_SCHEMA},
    }

    response, known_limit = _call_groq_with_backoff(
        client, model_name, messages, requested_max_tokens, response_format=response_format,
    )

    choice = response.choices[0]
    if choice.finish_reason == "length":
        raise RuntimeError(
            "The report reached the output limit. Set GROQ_MAX_COMPLETION_TOKENS to a higher value, then try again."
        )
    report = json.loads(choice.message.content or "{}")
    if report["urgency"] not in URGENCY_LEVELS or not report["report_markdown"].strip():
        raise ValueError("The AI returned an incomplete report.")
    if known_limit and known_limit < requested_max_tokens:
        report["note"] = (
            f"Your Groq plan currently allows only about {known_limit} output tokens per minute for "
            f"{model_name}, so this report was shortened to fit. Use the longer-report option, check "
            "console.groq.com/settings/limits, or upgrade your Groq plan, for more detail."
        )
    return report


def _section_instructions(headings) -> str:
    heading_list = "\n".join(headings)
    return f"""
You are a careful clinical-information assistant continuing one long patient report, one part at a time.
This is educational triage, not a diagnosis, a prescription, or a substitute for an in-person examination.
The patient's urgency level has already been determined and is included as "confirmed_urgency" in the
input — do not restate or re-derive it, just write consistently with it.

Write ONLY the following Markdown H2 heading(s), in this exact order, and nothing else — no preamble,
no other headings, no closing summary:
{heading_list}

Requirements for these sections:
- If "Possible causes to consider" is one of your headings, explain 3–6 plausible cause categories when
  the symptoms allow it. For each, explain why it may fit and what would support or argue against it.
  Clearly label them as possibilities, never conclusions.
- If a medication/treatment heading is included, discuss relevant medication classes or treatment
  categories only as topics to raise with a clinician/pharmacist. Never prescribe, recommend a dose,
  provide a dosing schedule, or tell the person to start, stop, or change any medicine.
- Include practical, low-risk comfort measures when appropriate, plus clear limits on when not to self-manage.
- Include specific red flags relevant to the stated symptoms, where relevant to your heading(s).
- Do not invent examination findings, test results, diagnoses, allergies, pregnancy status, or medical history.
- Do not be terse. Explain what needs assessment and why.
""".strip()


def generate_long_report(patient: dict, progress_callback=None) -> dict:
    """Return a longer report than a single call allows, by writing it in a few
    minute-spaced requests so each one gets a fresh per-minute output-token budget."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured. Add it, then restart Streamlit.")

    try:
        from groq import Groq
    except ModuleNotFoundError as error:
        raise RuntimeError("Groq package missing. Run: pip install --upgrade groq") from error

    def notify(message: str) -> None:
        if progress_callback:
            progress_callback(message)

    client = Groq(api_key=api_key)
    model_name = os.getenv("GROQ_HEALTH_MODEL", "qwen/qwen3.8-27b")
    known_limit = None

    urgency_instructions = (
        "You are a careful clinical-information assistant. Based solely on the intake data below, "
        "classify the situation into exactly one of the three categories in the supplied list:\n"
        '- "Emergency now": the described symptoms include or plausibly suggest a red-flag / '
        "life-threatening presentation (e.g. features consistent with stroke, heart attack, severe "
        "difficulty breathing, severe bleeding, anaphylaxis, overdose, sepsis, suicidal intent).\n"
        '- "Urgent same-day assessment": concerning enough to need a same-day clinician assessment, '
        "but nothing described suggests an immediate life-threatening emergency.\n"
        '- "Self-care": nothing described suggests urgent or emergency risk, and the symptoms can '
        "reasonably be monitored or self-managed while arranging routine care if needed.\n"
        "Base this only on what was reported — never on anything not stated. Return only JSON "
        "conforming to the schema — no explanation."
    )
    notify("Assessing urgency...")
    urgency_response, known_limit = _call_groq_with_backoff(
        client, model_name,
        [
            {"role": "system", "content": urgency_instructions},
            {"role": "user", "content": json.dumps(patient)},
        ],
        target_tokens=150,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "urgency_only", "strict": True, "schema": URGENCY_ONLY_SCHEMA},
        },
        known_limit=known_limit,
    )
    urgency_data = json.loads(urgency_response.choices[0].message.content or "{}")
    urgency = urgency_data.get("urgency")
    if urgency not in URGENCY_LEVELS:
        raise RuntimeError("The AI did not return a valid urgency level.")

    section_texts = []
    truncated_groups = []
    patient_with_urgency = {**patient, "confirmed_urgency": urgency}
    for index, headings in enumerate(SECTION_GROUPS):
        notify(f"Writing section {index + 1} of {len(SECTION_GROUPS)}...")
        response, known_limit = _call_groq_with_backoff(
            client, model_name,
            [
                {"role": "system", "content": _section_instructions(headings)},
                {"role": "user", "content": json.dumps(patient_with_urgency)},
            ],
            target_tokens=2500,
            known_limit=known_limit,
        )
        choice = response.choices[0]
        if choice.finish_reason == "length":
            truncated_groups.append(index + 1)
        text = (choice.message.content or "").strip()
        if not text:
            raise RuntimeError(f"The AI returned an empty section for: {', '.join(headings)}")
        section_texts.append(text)

    notify("Finishing up...")
    report = {"urgency": urgency, "report_markdown": "\n\n".join(section_texts)}
    note_parts = []
    if known_limit:
        note_parts.append(
            f"Generated across {len(SECTION_GROUPS) + 1} timed requests (~{known_limit} output "
            f"tokens/minute on your Groq plan for {model_name}) to fit more detail than one call allows."
        )
    if truncated_groups:
        note_parts.append(
            f"Section group(s) {', '.join(str(i) for i in truncated_groups)} were still cut off by the "
            "per-minute limit even after splitting — a higher Groq plan tier would complete them fully."
        )
    if note_parts:
        report["note"] = " ".join(note_parts)
    return report


def show_report(report: dict) -> None:
    st.subheader("AI health-information report")
    if report["urgency"] == "Emergency now":
        st.error(report["urgency"])
    elif report["urgency"] == "Urgent same-day assessment":
        st.warning(report["urgency"])
    else:
        st.success(report["urgency"])
    if report.get("note"):
        st.info(report["note"])
    st.markdown(report["report_markdown"])


# Embedded MPC logo, loaded from assets/mpc_logo.png (see _load_image_b64 above).
MPC_LOGO_B64 = _load_image_b64("mpc_logo.png")
MPC_LOGO_BYTES = base64.b64decode(MPC_LOGO_B64)

st.set_page_config(
    page_title="Med Pre-Consult(MPC)",
    page_icon=io.BytesIO(MPC_LOGO_BYTES),
    layout="wide",
)
initialize_database()

title_col1, title_col2 = st.columns([1, 8], vertical_alignment="center")
with title_col1:
    st.image(io.BytesIO(MPC_LOGO_BYTES), width=90)
with title_col2:
    st.title("Med Pre-Consult(MPC)")

st.caption("A detailed AI-generated explanation to help prepare for a clinical conversation.")
st.markdown(
    f"""
    <div style="display:flex;align-items:center;gap:8px;">
        <img src="data:image/png;base64,{ARCHLVE_LOGO_B64}"
             style="width:28px;height:28px;border-radius:50%;object-fit:cover;" />
        <span style="font-size:0.85rem;color:rgba(49,51,63,0.6);">Made by the Team AI Alchemist</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.warning("If there is immediate danger, severe chest pain, serious trouble breathing, a stroke-like symptom, overdose, severe bleeding, or a risk of self-harm, contact emergency services now.")

with st.sidebar:
    st.header("Groq configuration")
    if os.getenv("GROQ_API_KEY"):
        st.success("Groq API key detected")
    else:
        st.error(f"GROQ_API_KEY is not set. Add it to {ENV_PATH.name}, export it, or set it in Streamlit secrets.")

    with st.expander("Credits"):
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
                <img src="data:image/png;base64,{ARCHLVE_LOGO_B64}"
                     style="width:24px;height:24px;border-radius:50%;object-fit:cover;" />
                <span style="font-size:0.85rem;color:rgba(49,51,63,0.6);">Team AI Alchemist</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
                <img src="data:image/png;base64,{NIGHTMARE_AVATAR_B64}"
                     style="width:24px;height:24px;border-radius:50%;object-fit:cover;" />
                <span style="font-size:0.85rem;color:rgba(49,51,63,0.6);">Nishant</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for credit_name in ["Siddarth", "Vaibhav", "Satyajeet", "Vaibhavi", "Sameer"]:
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
                    <div style="width:24px;height:24px;border-radius:50%;background-color:#000;flex-shrink:0;"></div>
                    <span style="font-size:0.85rem;color:rgba(49,51,63,0.6);">{credit_name}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

with st.form("health_intake"):
    first, second = st.columns(2)
    with first:
        patient_name = st.text_input("Patient name (required only when saving)")
        age = st.number_input("Age", min_value=0, max_value=120, value=None, step=1)
        sex = st.selectbox("Sex assigned at birth (optional)", ["Not provided", "Female", "Male", "Intersex"])
        duration = st.text_input("When did this begin, and is it changing?")
        symptoms = st.text_area("Symptoms and main concern *", height=180, placeholder="Describe the symptom, location, pattern, triggers, and what makes it better or worse.")
    with second:
        conditions = st.text_area("Conditions, past procedures, and allergies")
        medicines = st.text_area("Current medicines, supplements, and recent medication changes")
        pregnancy = st.selectbox("Pregnancy possibility (optional)", ["Not provided", "No", "Yes", "Possibly"])
    goals = st.text_area("What detail would help you most?", placeholder="For example: possible causes, medicine interactions to discuss, tests, or self-care limits.")
    long_report = True
    consent = st.checkbox("I understand this is AI-generated health information, not medical diagnosis or prescribing.")
    storage_consent = st.checkbox("The patient consents to local storage of this intake and AI report for later search.")
    submitted = st.form_submit_button("Generate detailed AI report", type="primary")

if submitted:
    if not symptoms.strip() or not consent:
        st.error("Enter symptoms and confirm the acknowledgement first.")
    elif storage_consent and not patient_name.strip():
        st.error("Enter the patient name before saving a record.")
    else:
        patient = {
            "age": age,
            "sex_assigned_at_birth": sex,
            "pregnancy_possibility": pregnancy,
            "symptoms": symptoms.strip(),
            "duration_and_change": duration.strip(),
            "conditions_procedures_allergies": conditions.strip(),
            "current_medicines_supplements_changes": medicines.strip(),
            "information_requested": goals.strip(),
        }
        if long_report:
            status = st.empty()
            try:
                report = generate_long_report(patient, progress_callback=status.info)
            except RuntimeError as error:
                status.empty()
                st.error("The AI did not return a report, so no health guidance is being shown.")
                with st.expander("Technical details — use this to fix configuration"):
                    st.code(str(error), language=None)
                    st.caption(
                        "Common fixes: verify GROQ_API_KEY, check Groq account limits, run "
                        "`pip install --upgrade groq`, or set GROQ_HEALTH_MODEL to an available Groq model."
                    )
            else:
                status.empty()
                if storage_consent:
                    save_record(patient_name.strip(), patient, report)
                    st.success("Saved locally. Use the search section below to retrieve this record later.")
                show_report(report)
        else:
            with st.spinner("Preparing a detailed AI report..."):
                try:
                    report = generate_report(patient)
                except RuntimeError as error:
                    st.error("The AI did not return a report, so no health guidance is being shown.")
                    with st.expander("Technical details — use this to fix configuration"):
                        st.code(str(error), language=None)
                        st.caption(
                            "Common fixes: verify GROQ_API_KEY, check Groq account limits, run "
                            "`pip install --upgrade groq`, or set GROQ_HEALTH_MODEL to an available Groq model."
                        )
                else:
                    if storage_consent:
                        save_record(patient_name.strip(), patient, report)
                        st.success("Saved locally. Use the search section below to retrieve this record later.")
                    show_report(report)

st.divider()
st.subheader("Search saved patient records")
search_query = st.text_input("Search by patient name, symptoms, or report text")
if search_query.strip():
    rows = search_records(search_query)
    st.caption(f"{len(rows)} matching record(s)")
    for row in rows:
        with st.expander(f"{row['patient_name']} — {row['created_at'][:10]} — {row['urgency']}"):
            saved_patient = json.loads(row["patient_json"])
            st.write(f"**Symptoms:** {row['symptoms']}")
            st.write(f"**Age:** {saved_patient['age'] if saved_patient['age'] is not None else 'Not provided'}")
            show_report({"urgency": row["urgency"], "report_markdown": row["report_markdown"]})