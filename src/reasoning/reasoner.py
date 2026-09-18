"""
Clinical reasoning with Gemini LLM.
Uses gemini-2.0-flash-lite for speed.
"""

import os
import re
import json
import logging
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from src.reasoning.prompts import REASONING_PROMPT

logger = logging.getLogger(__name__)

VALID_ACTIONS = {"Immediate Visit", "Checkup", "No Action"}
DEFAULT_ACTION = "Checkup"


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if "text" in block:
                    parts.append(block["text"])
                elif "content" in block:
                    parts.append(block["content"])
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def get_llm():
    """Initialize Gemini LLM (fast lite model, longer timeout, retries)."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set. Check .env file.")

    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=api_key,
        response_mime_type="application/json",
        temperature=0.1,
        max_retries=2,
        request_timeout=90,
    )


def _clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\[[^\]]*\.txt[^\]]*\]', '', text)
    text = re.sub(r'\b\w+_\w+_\d{4}\.txt\b', '', text)
    text = re.sub(r'\b\w+\.txt\b', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _ensure_list(value, fallback=None) -> List:
    if fallback is None:
        fallback = []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        return [value]
    return fallback


def reason(
    symptoms: Dict[str, Any],
    ecg_findings: Dict[str, Any],
    evidence: List[Dict[str, Any]],
    past_records: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate triage action + narrative reason."""
    logger.info("Generating clinical reasoning")

    prompt = REASONING_PROMPT.format(
        symptoms=json.dumps(symptoms, indent=2) if symptoms else "No symptoms provided",
        ecg_findings=json.dumps(ecg_findings, indent=2) if ecg_findings else "No ECG provided",
        evidence=json.dumps(evidence, indent=2) if evidence else "No evidence retrieved",
    )

    try:
        llm = get_llm()
        response = llm.invoke([HumanMessage(content=prompt)])
        raw_text = _extract_text(response.content).strip()

        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            if raw_text.lower().startswith("json"):
                raw_text = raw_text[4:].lstrip()

        result = json.loads(raw_text)

        # Enforce valid action
        action = result.get("action", DEFAULT_ACTION)
        if action not in VALID_ACTIONS:
            action_lower = str(action).lower()
            if "immediate" in action_lower or "emergency" in action_lower:
                action = "Immediate Visit"
            elif "check" in action_lower or "follow" in action_lower:
                action = "Checkup"
            elif "no action" in action_lower or "none" in action_lower:
                action = "No Action"
            else:
                action = DEFAULT_ACTION
        result["action"] = action

        result["reason"] = _clean_text(result.get("reason", ""))
        if not result["reason"]:
            result["reason"] = _clean_text(result.get("explanation", ""))
        result["conditions"] = _ensure_list(result.get("conditions"))

        logger.info(f"Reasoning done. Action: {action}")
        return result

    except Exception as e:
        logger.error(f"Reasoning failed: {e}", exc_info=True)

        err_str = str(e).lower()
        if "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
            user_msg = "AI reasoning service is temporarily at capacity. Please try again shortly."
        elif "401" in err_str or "unauthorized" in err_str or "api key" in err_str:
            user_msg = "AI reasoning authentication failed. Check GOOGLE_API_KEY."
        elif "404" in err_str or "not found" in err_str:
            user_msg = "AI reasoning model unavailable."
        elif "timeout" in err_str or "deadline" in err_str or "504" in err_str:
            user_msg = "AI reasoning timed out. Please try again."
        else:
            user_msg = "AI reasoning service temporarily unavailable."

        return {
            "action": DEFAULT_ACTION,
            "reason": user_msg,
            "conditions": [],
        }