"""
CardioAgent - Triage app.
Structured symptoms + optional ECG -> triage action.

UPDATED:
  - Removed confidence metric from Summary.
  - Removed ECG image quality pass/fail banner.
  - Removed evidence retrieval scores.
  - Removed "Conditions" metric from Summary.
  - Summary now shows only: ECG Class and (optional) a plain status.
"""

import sys
import tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from src.pipeline import run_pipeline_safe


st.set_page_config(
    page_title="CardioAgent",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .main .block-container { max-width: 760px; padding-top: 2.5rem; padding-bottom: 3rem; }

        h2 { color: #0f2b46; font-weight: 700; letter-spacing: -0.5px; }
        h4 { color: #0f2b46; margin-top: 1.8rem; margin-bottom: 0.8rem; font-size: 1rem;
             text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }

        .action-badge {
            padding: 18px 26px;
            border-radius: 12px;
            font-weight: 700;
            font-size: 1.2rem;
            color: white;
            display: block;
            margin: 0.5rem 0 1.5rem 0;
            letter-spacing: 0.6px;
            text-align: center;
        }
        .action-immediate { background: linear-gradient(135deg, #b02a37, #8b1e29); }
        .action-checkup   { background: linear-gradient(135deg, #c45a0a, #a04a08); }
        .action-noaction  { background: linear-gradient(135deg, #1e7e34, #16632a); }

        .box {
            background: #f8fafc;
            border-left: 4px solid #1a5a8a;
            padding: 20px 24px;
            border-radius: 10px;
            line-height: 1.75;
            font-size: 1rem;
            color: #1e2d3d;
            margin: 0.4rem 0 1.5rem 0;
        }

        .pill {
            background: #ffffff;
            padding: 12px 18px;
            border-radius: 8px;
            border: 1px solid #e5e9ef;
            border-left: 3px solid #1a5a8a;
            margin: 6px 0;
            font-size: 0.95rem;
            color: #1e2d3d;
        }

        .evidence {
            background: #f8fafc;
            padding: 12px 16px;
            border-left: 3px solid #7a8899;
            border-radius: 6px;
            margin: 6px 0;
            font-size: 0.85rem;
            color: #4a5568;
        }

        div[data-testid="stMetricValue"] { font-size: 1.4rem; color: #0f2b46; }
        div[data-testid="stMetricLabel"] { font-size: 0.75rem; text-transform: uppercase;
                                           letter-spacing: 1px; color: #7a8899; }

        .stButton > button {
            background: #0f2b46;
            color: white;
            border: none;
            padding: 12px 24px;
            font-weight: 600;
            border-radius: 8px;
            font-size: 0.95rem;
            transition: background 0.2s;
        }
        .stButton > button:hover { background: #1a5a8a; color: white; }

        .stTextArea textarea {
            border-radius: 10px;
            border: 1px solid #e5e9ef;
            font-size: 0.95rem;
        }

        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("## CardioAgent")
st.caption("Clinical triage from symptoms and ECG.")

# ---------- Structured Inputs ----------
st.markdown("#### Patient Information")

col_a, col_b, col_c = st.columns(3)

with col_a:
    age = st.selectbox(
        "Age *",
        options=["Select"] + [str(i) for i in range(1, 121)],
    )

with col_b:
    sex = st.selectbox(
        "Sex *",
        options=["Select", "Male", "Female", "Other"],
    )

with col_c:
    duration_value = st.number_input(
        "Duration *",
        min_value=0,
        max_value=999,
        value=0,
        step=1,
    )

duration_unit = st.selectbox(
    "Duration unit *",
    options=["Select", "Hours", "Days", "Weeks", "Months"],
)

# ---------- Symptoms Checkboxes ----------
st.markdown("#### Symptoms *")

SYMPTOM_OPTIONS = [
    "Chest pain",
    "Chest pressure / tightness",
    "Pain radiating to left arm",
    "Pain radiating to jaw / neck",
    "Shortness of breath",
    "Sweating / diaphoresis",
    "Palpitations",
    "Dizziness / lightheadedness",
    "Fainting / syncope",
    "Fatigue",
    "Nausea / vomiting",
    "Swelling in legs / ankles",
    "Cough",
    "Wheezing",
    "Rapid heartbeat",
    "Slow heartbeat",
    "Anxiety / sense of doom",
    "Pain worse on exertion",
    "Pain relieved by rest",
]

selected_symptoms = []
cols = st.columns(3)
for i, symptom in enumerate(SYMPTOM_OPTIONS):
    with cols[i % 3]:
        if st.checkbox(symptom, key=f"symptom_{i}"):
            selected_symptoms.append(symptom)

other_symptoms = st.text_area(
    "Other symptoms / additional details",
    height=90,
    placeholder="e.g. pain started suddenly while resting, patient has history of diabetes and hypertension, currently on aspirin",
)

# ---------- ECG Upload ----------
ecg_file = st.file_uploader(
    "ECG image",
    type=["jpg", "jpeg", "png"],
)

run_clicked = st.button("Run Analysis", use_container_width=True)

if run_clicked:

    # ---------- Validation ----------
    errors = []
    if age == "Select":
        errors.append("Please select the patient's age.")
    if sex == "Select":
        errors.append("Please select the patient's sex.")
    if duration_unit == "Select":
        errors.append("Please select the duration unit.")
    if duration_value == 0 and duration_unit != "Select":
        errors.append("Please enter a duration greater than 0.")
    if not selected_symptoms and not other_symptoms.strip():
        errors.append("Please select at least one symptom or describe symptoms in 'Other'.")

    if errors:
        for err in errors:
            st.error(err)
        st.stop()

    # ---------- Build symptoms text ----------
    symptom_lines = []
    symptom_lines.append(
        f"{age} year old {sex.lower()} patient."
    )
    symptom_lines.append(
        f"Duration of symptoms: {duration_value} {duration_unit.lower()}."
    )
    if selected_symptoms:
        symptom_lines.append("Reported symptoms:")
        for s in selected_symptoms:
            symptom_lines.append(f"- {s}")
    if other_symptoms.strip():
        symptom_lines.append(f"Additional details: {other_symptoms.strip()}")

    symptoms_text = "\n".join(symptom_lines)

    # ---------- Save ECG to temp file ----------
    ecg_image_path = None
    if ecg_file is not None:
        suffix = Path(ecg_file.name).suffix
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(ecg_file.read())
        tmp.close()
        ecg_image_path = tmp.name

    progress_box = st.status("Starting analysis...", expanded=True)

    def show_progress(message):
        progress_box.write(message)

    with st.spinner("Running clinical triage..."):
        try:
            result = run_pipeline_safe(
                symptoms_text,
                ecg_image_path=ecg_image_path,
                past_records=None,
                progress_callback=show_progress,
            )
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()
        finally:
            progress_box.update(
                label="Analysis steps completed",
                state="complete",
                expanded=False,
            )

    st.markdown("---")

    # ---------- ACTION BADGE ----------
    action = result.get("action", "Checkup")
    action_class = {
        "Immediate Visit": "action-immediate",
        "Checkup":         "action-checkup",
        "No Action":       "action-noaction",
    }.get(action, "action-checkup")

    st.markdown(
        f'<div class="action-badge {action_class}">{action.upper()}</div>',
        unsafe_allow_html=True,
    )

    # ---------- REASON ----------
    reason = result.get("reason", "")
    if reason:
        st.markdown("#### Why this action")
        st.markdown(f'<div class="box">{reason}</div>', unsafe_allow_html=True)

    # ---------- ECG STATUS ----------
    ecg_meta = result.get("ecg_analysis", {})
    ecg_class = ecg_meta.get("predicted_class", "N/A")
    extraction_error = ecg_meta.get("extraction_error")

    if extraction_error:
        st.error(f"ECG image analysis failed: {extraction_error}")
    elif ecg_meta.get("source") == "image":
        # No pass/fail message — just silently note the source
        pass

    # ---------- SYMPTOMS (from pipeline) ----------
    symptoms = result.get("extracted_symptoms", {})
    if symptoms and symptoms.get("symptom_list"):
        st.markdown("#### Extracted Symptoms")
        cols = st.columns(3)
        with cols[0]:
            st.metric("Age", symptoms.get("age") or "—")
        with cols[1]:
            st.metric("Sex", symptoms.get("sex") or "—")
        with cols[2]:
            st.metric("Duration", symptoms.get("duration") or "—")

        st.markdown("**Symptoms identified:**")
        for s in symptoms.get("symptom_list", []):
            st.markdown(f"- {s}")

    # ---------- CONDITIONS ----------
    conditions = result.get("conditions", [])
    if conditions:
        st.markdown("#### Possible Conditions")
        for c in conditions:
            st.markdown(f'<div class="pill">{c}</div>', unsafe_allow_html=True)

    # ---------- EVIDENCE (sources only, no scores) ----------
    evidence = result.get("retrieved_evidence", [])
    if evidence:
        with st.expander("Guideline evidence used"):
            for e in evidence:
                source = e.get("source", "unknown")
                text = e.get("text", "")
                st.markdown(
                    f'<div class="evidence"><b>{source}</b><br>{text}...</div>',
                    unsafe_allow_html=True,
                )

    # ---------- RAW JSON (hidden expander) ----------
    with st.expander("Full result (JSON)"):
        st.json(result)