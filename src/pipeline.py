"""
CardioAgent Pipeline - Full end-to-end.
TRIAGE-ACTION version: accepts past_records, returns action + reason + conditions.

UPDATED:
  - Removed low-confidence downgrade. The reasoner's action is used as-is.
  - Removed "low-confidence (X%)" prepended text from reason.
  - The final action logged == the final action returned. No more mismatch.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Callable, Optional

sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from src.ecg.image_to_signal import extract_signal_from_image
    from src.ecg.interface import predict as ecg_predict
except ImportError as e:
    logger.warning(f"ECG module issue: {e}")

    def extract_signal_from_image(p, **kw):
        return {"signal": None, "success": False, "message": "ECG module not available"}

    def ecg_predict(sig, already_preprocessed=False, return_gradcam=True, return_model_used=False):
        result = ("NORM", 0.0, None)
        if return_model_used:
            return result + ("fallback",)
        return result

try:
    from src.rag.retriever import retrieve
except ImportError:
    def retrieve(q, top_k=5):
        return []

try:
    from src.reasoning.symptom_extractor import extract_symptoms
except ImportError:
    def extract_symptoms(t):
        return {"age": None, "sex": None, "symptom_list": [], "duration": None}

try:
    from src.reasoning.reasoner import reason
except ImportError:
    def reason(s, e, ev, past_records=None):
        return {
            "action": "Checkup",
            "reason": "Reasoning module unavailable.",
            "conditions": [],
        }


ECG_CLASS_TO_TERMS = {
    "MI":   "myocardial infarction",
    "STTC": "ST T wave changes ischemia",
    "CD":   "conduction disturbance bundle branch block",
    "HYP":  "hypertrophy",
}


def run_pipeline(
    raw_symptom_text,
    ecg_image_path=None,
    ecg_signal=None,
    past_records=None,
    top_k_retrieval=5,
    progress_callback: Optional[Callable[[str], None]] = None,
):
    """Full pipeline: symptoms + ECG + past records -> triage action."""
    def report(message: str) -> None:
        logger.info(message)
        if progress_callback is not None:
            progress_callback(message)

    logger.info("Pipeline start")

    result = {
        "pipeline_status": "running",
        "action": "Checkup",
        "reason": "",
        "conditions": [],
    }

    # ---------------------------------------------------------------
    # Step 1: Symptom extraction
    # ---------------------------------------------------------------
    report("Step 1: Extracting symptoms")
    try:
        symptoms = extract_symptoms(raw_symptom_text)
        report("Step 1 complete: Symptoms extracted")
    except Exception as e:
        logger.error(f"Symptom extraction failed: {e}")
        report(f"Step 1 warning: Symptom extraction failed: {e}")
        symptoms = {
            "age": None, "sex": None,
            "chief_complaint": None,
            "symptom_list": [], "duration": None,
            "radiation": None,
            "associated_symptoms": [],
            "severity_descriptors": [],
            "risk_factors": [],
        }
    result["extracted_symptoms"] = symptoms

    # ---------------------------------------------------------------
    # Step 2: ECG extraction
    # ---------------------------------------------------------------
    ecg_findings = {}
    ecg_meta = {"predicted_class": "N/A", "confidence": 0.0, "source": None}
    signal_to_use = None

    if ecg_image_path is not None:
        report("Step 2: Processing ECG image")
        ecg_meta["source"] = "image"
        try:
            extraction = extract_signal_from_image(ecg_image_path)
            if extraction.get("success"):
                signal_to_use = extraction["signal"]
                ecg_meta["extracted_leads"] = extraction.get("leads")
                ecg_meta["image_quality"] = extraction.get("quality")
                ecg_meta["lead_detection"] = extraction.get("lead_detection")
                ecg_meta["features"] = extraction.get("features")
                report(
                    f"Step 2 complete: ECG image converted to signal "
                    f"(shape {signal_to_use.shape})"
                )
            else:
                ecg_meta["extraction_error"] = (
                    extraction.get("error")
                    or extraction.get("message")
                    or "ECG image extraction failed without a diagnostic message."
                )
                inner_meta = extraction.get("metadata") or {}
                if inner_meta.get("validation_error"):
                    ecg_meta["validation_error"] = inner_meta["validation_error"]
                if inner_meta.get("lead_selection_error"):
                    ecg_meta["lead_selection_error"] = inner_meta["lead_selection_error"]
                logger.warning(f"Image extraction failed: {ecg_meta['extraction_error']}")
                report(f"Step 2 warning: ECG image processing failed: {ecg_meta['extraction_error']}")
        except Exception as e:
            logger.error(f"Image extraction crashed: {e}", exc_info=True)
            ecg_meta["extraction_error"] = str(e)
            report(f"Step 2 warning: ECG image processing crashed: {e}")

    elif ecg_signal is not None:
        report("Step 2: Using provided digital ECG signal")
        signal_to_use = ecg_signal
        ecg_meta["source"] = "signal"
        report("Step 2 complete: Digital ECG signal loaded")
    else:
        report("Step 2 skipped: No ECG image or signal provided")

    if signal_to_use is None:
        ecg_findings = {}
        ecg_meta.setdefault("predicted_class", "N/A")
        ecg_meta.setdefault("confidence", 0.0)

    # ---------------------------------------------------------------
    # Step 3: ECG classifier
    # ---------------------------------------------------------------
    if signal_to_use is not None:
        report("Step 3: Running ECG classifier")
        try:
            ecg_result = ecg_predict(signal_to_use, return_model_used=True)

            if isinstance(ecg_result, tuple):
                predicted_class = ecg_result[0] if len(ecg_result) > 0 else "NORM"
                confidence = ecg_result[1] if len(ecg_result) > 1 else 0.0
                gradcam_map = ecg_result[2] if len(ecg_result) > 2 else None
                model_used = ecg_result[3] if len(ecg_result) > 3 else None
            else:
                predicted_class = ecg_result.get("class", "NORM")
                confidence = ecg_result.get("confidence", 0.0)
                gradcam_map = ecg_result.get("gradcam")
                model_used = ecg_result.get("model_used")

            ecg_findings = {
                "class": predicted_class,
                "confidence": float(confidence),
            }
            ecg_meta["predicted_class"] = predicted_class
            ecg_meta["confidence"] = float(confidence)
            ecg_meta["model_used"] = model_used

            report(f"Step 3 complete: ECG classified as {predicted_class}")
        except Exception as e:
            logger.error(f"ECG model crashed: {e}", exc_info=True)
            ecg_meta["classifier_error"] = str(e)
            report(f"Step 3 warning: ECG classifier failed: {e}")
    else:
        report("Step 3 skipped: No ECG signal available")

    result["ecg_analysis"] = ecg_meta

    # ---------------------------------------------------------------
    # Step 4: Build RAG query
    # ---------------------------------------------------------------
    query_parts = list(symptoms.get("symptom_list", []))[:3]
    ecg_class = ecg_findings.get("class")
    if ecg_class and ecg_class != "NORM":
        query_parts.append(ECG_CLASS_TO_TERMS.get(ecg_class, ecg_class))
    query = " ".join(query_parts) if query_parts else raw_symptom_text[:200]
    report("Step 4: Building evidence search query")
    logger.info(f"RAG query: {query[:120]}")

    # ---------------------------------------------------------------
    # Step 5: RAG retrieval
    # ---------------------------------------------------------------
    report("Step 5: Retrieving clinical evidence")
    try:
        evidence = retrieve(query, top_k=top_k_retrieval)
    except Exception as e:
        logger.error(f"RAG retrieval failed: {e}")
        report(f"Step 5 warning: Evidence retrieval failed: {e}")
        evidence = []
    else:
        report(f"Step 5 complete: Retrieved {len(evidence)} evidence items")

    result["retrieved_evidence"] = [
        {
            "source": e.get("source"),
            "score": e.get("score"),
            "text": (e.get("chunk", "") or "")[:300],
        }
        for e in evidence[:3]
    ]

    # ---------------------------------------------------------------
    # Step 6: Clinical reasoning
    # ---------------------------------------------------------------
    report("Step 6: Generating clinical triage")
    try:
        reasoning = reason(symptoms, ecg_findings, evidence, past_records=past_records)
        result["action"] = reasoning.get("action", "Checkup")
        result["reason"] = reasoning.get("reason", "")
        result["conditions"] = reasoning.get("conditions", [])
    except Exception as e:
        logger.error(f"Reasoning failed: {e}")
        report(f"Step 6 warning: Clinical reasoning failed: {e}")
        result["reason"] = "Clinical reasoning service unavailable."

        fb_class = ecg_findings.get("class", "NORM")
        fb_conf = ecg_findings.get("confidence", 0.0)

        if fb_class == "MI" and fb_conf >= 0.5:
            result["action"] = "Immediate Visit"
            result["reason"] = (
                "ECG finding suggests a possible myocardial infarction. "
                "Immediate emergency evaluation is advised."
            )
        elif fb_class in ("STTC", "CD", "HYP") and fb_conf >= 0.5:
            result["action"] = "Checkup"
            result["reason"] = (
                "ECG finding suggests a possible abnormality. "
                "Clinical follow-up is recommended."
            )
        else:
            result["action"] = "Checkup"

    # ---------------------------------------------------------------
    # Final triage action — uses the reasoner's decision directly.
    # No confidence-based downgrade, no prepended warning text.
    # ---------------------------------------------------------------
    if result["action"] not in {"Immediate Visit", "Checkup", "No Action"}:
        result["action"] = "Checkup"

    report(f"Step 6 complete: Triage action is {result['action']}")

    result["pipeline_status"] = "complete"
    logger.info(f"Pipeline complete. Action: {result['action']}")
    return result


def run_pipeline_safe(
    raw_symptom_text,
    ecg_image_path=None,
    ecg_signal=None,
    past_records=None,
    progress_callback: Optional[Callable[[str], None]] = None,
):
    """Safe wrapper that never crashes."""
    try:
        return run_pipeline(
            raw_symptom_text,
            ecg_image_path,
            ecg_signal,
            past_records,
            progress_callback=progress_callback,
        )
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return {
            "pipeline_status": "error",
            "action": "Checkup",
            "reason": f"Pipeline error: {str(e)}",
            "conditions": [],
            "ecg_analysis": {"predicted_class": "N/A", "confidence": 0.0},
            "extracted_symptoms": {},
            "retrieved_evidence": [],
        }