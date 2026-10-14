"""
CardioAgent - Triage app.
Structured symptoms + optional ECG -> triage action.

Includes:
- Light and dark mode toggle
- Patient intake and symptom selection
- ECG upload and triage results
- Guideline evidence (top-5 retrieved passages)
- Grad-CAM visualization without displaying confidence
"""
import os
import html
import json
import sys
import tempfile
from pathlib import Path
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))
load_dotenv()

from src.pipeline import run_pipeline_safe


st.set_page_config(
    page_title="CardioAgent",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# THEME
# ---------------------------------------------------------------------------

THEMES = {
    "light": {
        "scheme": "light",
        "ink": "#10243a",
        "slate": "#5b6b7c",
        "line": "#dfe5ec",
        "field-border": "#b9c5d2",
        "canvas": "#f3f6f9",
        "surface": "#ffffff",
        "accent": "#0e7c86",
        "accent-soft": "#e3f3f4",
        "chip-fg": "#08565d",
        "urgent": "#b3261e",
        "soon": "#b35a00",
        "clear": "#1b7a3d",
        "stale-bg": "#fff3e2",
        "stale-fg": "#b35a00",
        "evid-bg": "#f7f9fb",
        "muted": "#9aa7b5",
        "btn-bg": "#10243a",
        "btn-fg": "#ffffff",
        "btn-hover": "#0e7c86",
        "disabled-bg": "#c9d2dc",
        "disabled-fg": "#ffffff",
        "brand-mark": "#10243a",
    },
    "dark": {
        "scheme": "dark",
        "ink": "#e8eef5",
        "slate": "#9fb0c1",
        "line": "#263646",
        "field-border": "#4a6178",
        "canvas": "#0c1520",
        "surface": "#142130",
        "accent": "#35bfcb",
        "accent-soft": "#12383d",
        "chip-fg": "#8be0e7",
        "urgent": "#c93a32",
        "soon": "#c2630a",
        "clear": "#23803f",
        "stale-bg": "#3a2a12",
        "stale-fg": "#f0b45f",
        "evid-bg": "#101c29",
        "muted": "#6f8296",
        "btn-bg": "#35bfcb",
        "btn-fg": "#04242a",
        "btn-hover": "#5fd0da",
        "disabled-bg": "#26364a",
        "disabled-fg": "#6f8296",
        "brand-mark": "#1b3a4e",
    },
}

# ---------------------------------------------------------------------------
# SYMPTOMS
# ---------------------------------------------------------------------------

OTHER_LABEL = "Other symptoms"

SYMPTOM_OPTIONS = [
    "Chest pain",
    "Chest pressure",
    "Chest tightness",
    "Pain radiating to left arm",
    "Pain radiating to jaw",
    "Pain radiating to neck",
    "Shortness of breath",
    "Sweating",
    "Palpitations",
    "Dizziness",
    "Lightheadedness",
    "Fainting",
    "Fatigue",
    "Nausea",
    "Vomiting",
    "Swelling in legs",
    "Swelling in ankles",
    "Cough",
    "Wheezing",
    "Rapid heartbeat",
    "Slow heartbeat",
    "Anxiety",
    "Sense of doom",
    "Pain worse on exertion",
    "Pain relieved by rest",
    OTHER_LABEL,
]

DURATION_UNITS = ["Hours", "Days", "Weeks", "Months"]

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

head_l, head_r = st.columns([5, 1.3], vertical_alignment="center")

with head_r:
    dark_mode = st.toggle("Dark mode", key="dark_mode")

tokens = THEMES["dark" if dark_mode else "light"]
root_vars = ":root{" + ";".join(
    f"--{key}:{value}" for key, value in tokens.items()
) + "}"

STATIC_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

:root { color-scheme: var(--scheme); }

html, body, .stApp, [class*="css"] {
    font-family: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
.stApp, [data-testid="stAppViewContainer"] {
    background: var(--canvas);
    color: var(--ink);
}
.main .block-container {
    max-width: 1180px;
    padding-top: 1.4rem;
    padding-bottom: 3rem;
}
#MainMenu, footer, header[data-testid="stHeader"] {
    visibility: hidden;
    height: 0;
}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
.stApp label p,
[data-testid="stWidgetLabel"] p {
    color: var(--ink);
}
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: var(--slate);
}
.brand { display:flex; align-items:center; gap:12px; }
.brand-mark {
    width:40px; height:40px; border-radius:11px;
    background:var(--brand-mark);
    display:flex; align-items:center; justify-content:center;
}
.brand-name {
    font-size:1.3rem; font-weight:700;
    letter-spacing:-0.3px; line-height:1.1; color:var(--ink);
}
.brand-sub { font-size:0.84rem; color:var(--slate); }
.divider { height:1px; background:var(--line); margin:0.6rem 0 1.4rem; }
.sec-title {
    font-size:1.02rem; font-weight:600;
    margin:1.4rem 0 0.15rem; color:var(--ink);
}
.sec-hint { font-size:0.85rem; color:var(--slate); margin-bottom:0.6rem; }

div[data-testid="stNumberInputContainer"],
div[data-testid="stNumberInputContainer"] div[data-baseweb],
div[data-testid="stNumberInputContainer"] div[data-baseweb] > div,
div[data-baseweb="input"],
div[data-baseweb="base-input"],
div[data-baseweb="select"] > div,
div[data-baseweb="select"] > div > div,
.stTextArea textarea,
.stTextArea div[data-baseweb] {
    background:var(--surface) !important;
    background-color:var(--surface) !important;
    border-radius:10px !important;
}
div[data-baseweb="input"],
div[data-baseweb="select"] > div,
.stTextArea div[data-baseweb] {
    border:1px solid var(--field-border) !important;
}
input, textarea {
    color:var(--ink) !important;
    -webkit-text-fill-color:var(--ink) !important;
    background:transparent !important;
    font-size:1rem !important;
    font-weight:500;
}
input::placeholder, textarea::placeholder {
    color:var(--slate) !important;
    -webkit-text-fill-color:var(--slate) !important;
    opacity:.8;
    font-weight:400;
}
div[data-baseweb="input"]:focus-within,
div[data-baseweb="select"] > div:focus-within,
.stTextArea textarea:focus {
    border-color:var(--accent) !important;
    box-shadow:0 0 0 3px var(--accent-soft) !important;
}
div[data-baseweb="select"] svg,
div[data-testid="stNumberInputContainer"] svg {
    fill:var(--slate);
    color:var(--slate);
}
div[data-testid="stNumberInputContainer"] button {
    background:var(--surface) !important;
    color:var(--ink) !important;
    border:1px solid var(--field-border) !important;
}
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
div[data-baseweb="menu"],
div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] li {
    background:var(--surface) !important;
    color:var(--ink) !important;
}
div[data-baseweb="popover"] li:hover,
div[data-baseweb="popover"] li[aria-selected="true"] {
    background:var(--accent-soft) !important;
}
span[data-baseweb="tag"] {
    background:var(--accent-soft) !important;
    color:var(--chip-fg) !important;
    border-radius:6px !important;
    font-weight:500;
}
span[data-baseweb="tag"] span { color:var(--chip-fg) !important; }
span[data-baseweb="tag"] svg { fill:var(--chip-fg); }

div[role="radiogroup"] { gap:8px; flex-wrap:wrap; }
div[role="radiogroup"] label {
    background:var(--surface) !important;
    border:1px solid var(--field-border);
    padding:7px 14px;
    border-radius:10px;
    margin:0 !important;
}
div[role="radiogroup"] label:has(input:checked) {
    border-color:var(--accent);
    background:var(--accent-soft) !important;
}
div[role="radiogroup"] label p {
    font-size:0.9rem;
    font-weight:500;
    color:var(--ink);
}
.stButton > button {
    background:var(--btn-bg);
    color:var(--btn-fg);
    border:none;
    padding:0.85rem 1.2rem;
    font-weight:600;
    font-size:1rem;
    border-radius:12px;
}
.stButton > button p { color:var(--btn-fg); }
.stButton > button:hover:not(:disabled) {
    background:var(--btn-hover);
    color:var(--btn-fg);
}
.stButton > button:disabled {
    background:var(--disabled-bg);
    color:var(--disabled-fg);
}
section[data-testid="stFileUploaderDropzone"] {
    background:var(--surface);
    border:1.5px dashed var(--muted);
    border-radius:12px;
}
section[data-testid="stFileUploaderDropzone"] * { color:var(--slate); }
.card {
    background:var(--surface);
    border:1px solid var(--line);
    border-radius:14px;
    padding:20px 22px;
    margin-bottom:14px;
}
.card-title {
    font-size:0.84rem;
    font-weight:600;
    color:var(--slate);
    margin-bottom:10px;
}
.pt-line {
    font-size:1.35rem;
    font-weight:600;
    letter-spacing:-0.3px;
    color:var(--ink);
}
.pt-line.empty { color:var(--muted); font-weight:500; }
.pt-sub { font-size:0.92rem; color:var(--slate); margin-top:2px; }
.chips { display:flex; flex-wrap:wrap; gap:6px; margin-top:4px; }
.chip {
    background:var(--accent-soft);
    color:var(--chip-fg);
    font-size:0.82rem;
    font-weight:500;
    padding:4px 10px;
    border-radius:6px;
}
.sub-label { font-size:0.8rem; color:var(--slate); margin:14px 0 6px; }
.other-text { font-size:0.9rem; color:var(--ink); line-height:1.5; }
.empty-note { font-size:0.9rem; color:var(--muted); }
.result-head {
    margin:2rem 0 0.8rem;
    display:flex;
    align-items:baseline;
    gap:12px;
    flex-wrap:wrap;
}
.result-head h3 {
    margin:0;
    font-size:1.15rem;
    font-weight:600;
    color:var(--ink);
}
.stale {
    font-size:0.8rem;
    font-weight:500;
    color:var(--stale-fg);
    background:var(--stale-bg);
    padding:3px 10px;
    border-radius:999px;
}
.action-badge {
    display:flex;
    align-items:center;
    padding:22px 26px;
    border-radius:14px;
    color:#fff;
    font-weight:700;
    font-size:1.5rem;
}
.action-badge small {
    display:block;
    font-size:0.82rem;
    font-weight:500;
    opacity:.85;
}
.action-immediate { background:var(--urgent); }
.action-checkup { background:var(--soon); }
.action-noaction { background:var(--clear); }
.reason {
    background:var(--surface);
    color:var(--ink);
    border:1px solid var(--line);
    border-left:4px solid var(--accent);
    padding:18px 22px;
    border-radius:12px;
    line-height:1.7;
    font-size:0.98rem;
    white-space:pre-wrap;
    margin-top:12px;
}
.cond {
    background:var(--surface);
    color:var(--ink);
    border:1px solid var(--line);
    border-radius:10px;
    padding:11px 16px;
    margin-bottom:8px;
    font-size:0.95rem;
}
.evidence {
    background:var(--evid-bg);
    color:var(--slate);
    border-left:3px solid var(--muted);
    border-radius:6px;
    padding:10px 14px;
    margin:6px 0;
    font-size:0.85rem;
    line-height:1.55;
}
.evidence b { color:var(--ink); }
.evidence .score {
    float:right;
    font-size:0.78rem;
    color:var(--muted);
}
[data-testid="stExpander"] {
    background:var(--surface);
    border:1px solid var(--line);
    border-radius:12px;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * { color:var(--ink); }
.disclaimer {
    font-size:0.8rem;
    color:var(--slate);
    margin-top:2rem;
    text-align:center;
}
"""

st.markdown(f"<style>{root_vars}{STATIC_CSS}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------

with head_l:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-mark">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none"
                     stroke="#ffffff" stroke-width="2"
                     stroke-linecap="round" stroke-linejoin="round">
                    <path d="M3 12h4l2-6 4 12 2-6h6"/>
                </svg>
            </div>
            <div>
                <div class="brand-name">CardioAgent</div>
                <div class="brand-sub">Clinical triage from symptoms and ECG</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

left, right = st.columns([1.15, 1], gap="large")

# ---------------------------------------------------------------------------
# LEFT: PATIENT INTAKE
# ---------------------------------------------------------------------------

with left:
    st.markdown('<div class="sec-title">Patient</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])

    with c1:
        age = st.number_input(
            "Age (years) *",
            min_value=0,
            max_value=120,
            value=None,
            step=1,
            placeholder="e.g. 54",
        )

    with c2:
        sex = st.radio(
            "Sex *",
            options=["Male", "Female", "Other"],
            index=None,
            horizontal=True,
        )

    st.markdown('<div class="sec-title">Onset</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-hint">How long have the symptoms been present?</div>',
        unsafe_allow_html=True,
    )

    d1, d2 = st.columns([1, 2])

    with d1:
        duration_value = st.number_input(
            "Duration *",
            min_value=0,
            max_value=999,
            value=None,
            step=1,
            placeholder="e.g. 2",
        )

    with d2:
        duration_unit = st.radio(
            "Unit *",
            options=DURATION_UNITS,
            index=None,
            horizontal=True,
        )

    st.markdown('<div class="sec-title">Symptoms</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-hint">Open the list or type to search. Choose '
        '"Other symptoms" to describe anything not listed.</div>',
        unsafe_allow_html=True,
    )

    selected_all = st.multiselect(
        "Symptoms *",
        options=SYMPTOM_OPTIONS,
        placeholder="Search or select symptoms",
        label_visibility="collapsed",
    )

    other_enabled = OTHER_LABEL in selected_all
    selected_symptoms = [s for s in selected_all if s != OTHER_LABEL]

    other_text = ""
    if other_enabled:
        other_text = st.text_area(
            "Describe other symptoms",
            height=100,
            placeholder="Describe any additional symptoms or relevant context",
        ).strip()

    st.markdown('<div class="sec-title">ECG</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-hint">Optional. Upload a photo or scan of the ECG strip.</div>',
        unsafe_allow_html=True,
    )

    ecg_file = st.file_uploader(
        "ECG image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if ecg_file is not None:
        st.image(ecg_file, caption=ecg_file.name, use_container_width=True)

# ---------------------------------------------------------------------------
# LIVE READINESS
# ---------------------------------------------------------------------------

has_age = age is not None
has_sex = sex is not None
has_duration = (
    duration_value is not None
    and duration_value > 0
    and duration_unit is not None
)
has_symptoms = bool(selected_symptoms) or bool(other_text)
ready = has_age and has_sex and has_duration and has_symptoms

# ---------------------------------------------------------------------------
# RIGHT: PATIENT SUMMARY
# ---------------------------------------------------------------------------

with right:
    if has_age or has_sex:
        parts = []
        if has_age:
            parts.append(f"{int(age)} years")
        if has_sex:
            parts.append(sex)
        pt_html = f'<div class="pt-line">{html.escape(" · ".join(parts))}</div>'
    else:
        pt_html = (
            '<div class="pt-line empty">'
            'Patient details will appear here</div>'
        )

    if duration_value and duration_unit:
        pt_html += (
            f'<div class="pt-sub">Symptoms for {int(duration_value)} '
            f'{duration_unit.lower()}</div>'
        )

    if selected_symptoms:
        chips = "".join(
            f'<span class="chip">{html.escape(s)}</span>'
            for s in selected_symptoms
        )
        chips_html = f'<div class="chips">{chips}</div>'
    else:
        chips_html = '<div class="empty-note">No symptoms selected yet</div>'

    if other_text:
        chips_html += (
            '<div class="sub-label">Other symptoms</div>'
            f'<div class="other-text">{html.escape(other_text)}</div>'
        )

    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">Patient summary</div>
            {pt_html}
        </div>
        <div class="card">
            <div class="card-title">Symptoms ({len(selected_symptoms)} selected)</div>
            {chips_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    run_clicked = st.button(
        "Run analysis",
        use_container_width=True,
        disabled=not ready,
    )

    if not ready:
        st.caption(
            "Fill in age, sex, duration and at least one symptom "
            "to enable the analysis."
        )

# ---------------------------------------------------------------------------
# INPUT SIGNATURE
# ---------------------------------------------------------------------------

current_signature = (
    age,
    sex,
    duration_value,
    duration_unit,
    tuple(selected_symptoms),
    other_text,
    ecg_file.name if ecg_file is not None else None,
)

# ---------------------------------------------------------------------------
# RUN PIPELINE
# ---------------------------------------------------------------------------

if run_clicked and ready:
    symptom_lines = [
        f"{int(age)} year old {sex.lower()} patient.",
        f"Duration of symptoms: {int(duration_value)} {duration_unit.lower()}.",
    ]

    if selected_symptoms:
        symptom_lines.append("Reported symptoms:")
        symptom_lines.extend(f"- {s}" for s in selected_symptoms)

    if other_text:
        symptom_lines.append(f"Additional details: {other_text}")

    symptoms_text = "\n".join(symptom_lines)

    ecg_image_path = None

    if ecg_file is not None:
        suffix = Path(ecg_file.name).suffix or ".png"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(ecg_file.getvalue())
            ecg_image_path = tmp.name

    progress_box = st.status("Starting analysis...", expanded=True)

    def show_progress(message):
        progress_box.write(message)

    try:
        result = run_pipeline_safe(
            symptoms_text,
            ecg_image_path=ecg_image_path,
            past_records=None,
            progress_callback=show_progress,
        )
        progress_box.update(
            label="Analysis complete",
            state="complete",
            expanded=False,
        )
    except Exception as e:
        progress_box.update(
            label="Analysis failed",
            state="error",
            expanded=True,
        )
        st.error(f"Pipeline error: {e}")
        st.stop()
    finally:
        if ecg_image_path and os.path.exists(ecg_image_path):
            os.unlink(ecg_image_path)

    st.session_state["result"] = result
    st.session_state["result_signature"] = current_signature

# ---------------------------------------------------------------------------
# RESULT DISPLAY
# ---------------------------------------------------------------------------

result = st.session_state.get("result")

if result:
    is_stale = (
        st.session_state.get("result_signature") != current_signature
    )
    stale_html = (
        '<span class="stale">Inputs changed since this result</span>'
        if is_stale else ""
    )

    st.markdown(
        f'<div class="result-head"><h3>Triage result</h3>{stale_html}</div>',
        unsafe_allow_html=True,
    )

    res_left, res_right = st.columns([1.15, 1], gap="large")

    action = result.get("action", "Checkup")
    action_class = {
        "Immediate Visit": "action-immediate",
        "Checkup": "action-checkup",
        "No Action": "action-noaction",
    }.get(action, "action-checkup")

    with res_left:
        st.markdown(
            f'<div class="action-badge {action_class}"><div>'
            f'<small>Recommended action</small>'
            f'{html.escape(action.upper())}</div></div>',
            unsafe_allow_html=True,
        )

        reason = result.get("reason", "")
        if reason:
            st.markdown(
                '<div class="sec-title">Why this action</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="reason">{html.escape(str(reason))}</div>',
                unsafe_allow_html=True,
            )

        ecg_meta = result.get("ecg_analysis", {}) or {}

        extraction_error = ecg_meta.get("extraction_error")
        if extraction_error:
            st.error(f"ECG image analysis failed: {extraction_error}")

    with res_right:
        symptoms = result.get("extracted_symptoms", {})

        if symptoms and symptoms.get("symptom_list"):
            st.markdown(
                '<div class="sec-title">Extracted symptoms</div>',
                unsafe_allow_html=True,
            )

            m1, m2, m3 = st.columns(3)

            with m1:
                st.metric("Age", symptoms.get("age") or "—")
            with m2:
                st.metric("Sex", symptoms.get("sex") or "—")
            with m3:
                st.metric("Duration", symptoms.get("duration") or "—")

            chips = "".join(
                f'<span class="chip">{html.escape(str(s))}</span>'
                for s in symptoms.get("symptom_list", [])
            )

            st.markdown(
                f'<div class="chips" style="margin-top:8px">{chips}</div>',
                unsafe_allow_html=True,
            )

        conditions = result.get("conditions", [])

        if conditions:
            st.markdown(
                '<div class="sec-title">Possible conditions</div>',
                unsafe_allow_html=True,
            )

            for condition in conditions:
                st.markdown(
                    f'<div class="cond">{html.escape(str(condition))}</div>',
                    unsafe_allow_html=True,
                )

    # -----------------------------------------------------------------------
    # GUIDELINE EVIDENCE (retrieved passages)
    # -----------------------------------------------------------------------

    evidence_items = result.get("retrieved_evidence", []) or []

    st.markdown(
        '<div class="sec-title">Guideline evidence</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Passages retrieved from the cardiology knowledge base and "
        "provided to the reasoning model."
    )

    if evidence_items:
        retrieval_query = result.get("retrieval_query")
        if retrieval_query:
            st.caption(f"Search query: {retrieval_query}")

        with st.expander(
            f"Retrieved passages ({len(evidence_items)})",
            expanded=True,
        ):
            for i, item in enumerate(evidence_items, start=1):
                source = html.escape(
                    str(item.get("source") or "Unknown source")
                )
                text = html.escape(str(item.get("text") or ""))

                score = item.get("score")
                try:
                    score_html = (
                        f'<span class="score">Similarity: '
                        f'{float(score):.3f}</span>'
                    )
                except (TypeError, ValueError):
                    score_html = ""

                st.markdown(
                    f'<div class="evidence">{score_html}'
                    f'<b>{i}. {source}</b><br>{text}</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.info(
            "No evidence passages were returned. Check that the "
            "retriever loaded correctly and the knowledge base index exists."
        )

    # -----------------------------------------------------------------------
    # GRAD-CAM VISUALIZATION
    # -----------------------------------------------------------------------

    ecg_meta = result.get("ecg_analysis", {}) or {}
    gradcam_map = ecg_meta.get("gradcam_map")
    ecg_signal = ecg_meta.get("signal")
    predicted_class = ecg_meta.get("predicted_class", "N/A")

    st.markdown(
        '<div class="sec-title">ECG Explainability — Grad-CAM</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "The coloured waveform shows the relative time points that "
        "influenced the model prediction. It does not confirm a diagnosis."
    )

    if gradcam_map is None:
        st.info(
            "Grad-CAM is unavailable. Check that the ECG prediction "
            "function returns a Grad-CAM map."
        )

    elif ecg_signal is None:
        st.info(
            "Grad-CAM is available, but the ECG waveform was not returned "
            "by the pipeline."
        )

    else:
        try:
            signal = np.asarray(ecg_signal, dtype=float)
            cam = np.asarray(gradcam_map, dtype=float).squeeze()

            if signal.ndim != 2:
                raise ValueError(
                    f"Expected a 2D ECG signal; received {signal.shape}."
                )

            # Convert (8, samples) to (samples, 8), if needed.
            if signal.shape[1] != 8 and signal.shape[0] == 8:
                signal = signal.T

            if signal.shape[1] != 8:
                raise ValueError(
                    f"Expected 8 leads; received shape {signal.shape}."
                )

            if cam.ndim != 1:
                raise ValueError(
                    f"Expected a 1D Grad-CAM map; received {cam.shape}."
                )

            if len(cam) != len(signal):
                raise ValueError(
                    f"ECG has {len(signal)} samples, but Grad-CAM has "
                    f"{len(cam)} values."
                )

            if (
                not np.isfinite(signal).all()
                or not np.isfinite(cam).all()
            ):
                raise ValueError(
                    "ECG or Grad-CAM contains invalid numeric values."
                )

            # Normalize for display. This does not change the model prediction.
            cam_min = float(cam.min())
            cam_max = float(cam.max())

            if cam_max > cam_min:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = np.zeros_like(cam)

            lead_names = [
                "I", "II", "V1", "V2",
                "V3", "V4", "V5", "V6",
            ]

            sampling_rate = 100
            time = np.arange(len(signal)) / sampling_rate

            cmap = plt.get_cmap("jet")
            norm = Normalize(vmin=0.0, vmax=1.0)

            fig, axes = plt.subplots(
                nrows=8,
                ncols=1,
                figsize=(15, 11),
                sharex=True,
            )

            fig.subplots_adjust(
                left=0.08,
                right=0.88,
                top=0.93,
                bottom=0.07,
                hspace=0.18,
            )

            for lead_index, ax in enumerate(axes):
                waveform = signal[:, lead_index]

                points = np.column_stack((time, waveform))

                segments = np.stack(
                    [points[:-1], points[1:]],
                    axis=1,
                )

                # Colour the waveform by the Grad-CAM value at each time.
                line = LineCollection(
                    segments,
                    cmap=cmap,
                    norm=norm,
                    linewidth=1.0,
                )

                line.set_array(
                    (cam[:-1] + cam[1:]) / 2.0
                )

                ax.add_collection(line)

                low = float(waveform.min())
                high = float(waveform.max())
                amplitude_range = high - low

                margin = (
                    amplitude_range * 0.15
                    if amplitude_range > 1e-8
                    else 1.0
                )

                ax.set_xlim(time[0], time[-1])
                ax.set_ylim(
                    low - margin,
                    high + margin,
                )

                ax.set_ylabel(
                    lead_names[lead_index],
                    rotation=0,
                    labelpad=20,
                )

                ax.grid(True, alpha=0.2)
                ax.tick_params(axis="both", labelsize=8)

            axes[-1].set_xlabel("Time (seconds)")
            axes[-1].set_xticks(
                np.arange(0, time[-1] + 0.001, 1.0)
            )

            fig.suptitle(
                "ECG Waveforms with Grad-CAM Overlay",
                fontsize=14,
                fontweight="bold",
            )

            scalar_map = plt.cm.ScalarMappable(
                norm=norm,
                cmap=cmap,
            )
            scalar_map.set_array([])

            colorbar = fig.colorbar(
                scalar_map,
                ax=list(axes),
                location="right",
                fraction=0.025,
                pad=0.02,
            )

            colorbar.set_label(
                "Relative Grad-CAM importance"
            )

            colorbar.set_ticks([0.0, 0.5, 1.0])
            colorbar.set_ticklabels(
                ["Lower", "Medium", "Higher"]
            )

            st.markdown(
                f"**Predicted ECG class:** "
                f"{html.escape(str(predicted_class))}"
            )

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            with st.expander("How to interpret Grad-CAM"):
                st.write(
                    "Brighter colours represent higher relative Grad-CAM "
                    "importance; darker colours represent lower importance."
                )
                st.write(
                    "The same temporal map is displayed across all eight "
                    "leads. It is not an independent explanation for each lead."
                )
                st.write(
                    "Grad-CAM shows which time regions influenced the model's "
                    "prediction. It cannot establish that the predicted class "
                    "is medically correct."
                )

        except (ValueError, TypeError, IndexError) as exc:
            st.warning(
                f"Grad-CAM visualization could not be displayed: {exc}"
            )