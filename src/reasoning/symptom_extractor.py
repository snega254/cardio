"""
Symptom extractor - converts free text to structured symptoms.
Uses gemini-2.0-flash-lite for speed.
"""

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import json
import logging
from typing import Dict, Any

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


SYMPTOM_EXTRACTION_PROMPT = """You are a clinical information extractor.

Extract ONLY what is stated in the text below. Do NOT infer or add information.

Return ONLY valid JSON with these exact keys:
- "age": integer or null
- "sex": "Male" | "Female" | null
- "chief_complaint": string or null (primary symptom, e.g. "chest pain")
- "symptom_list": list of strings (all symptoms mentioned)
- "duration": string or null (e.g., "2 hours")
- "radiation": string or null (e.g., "to left arm")
- "associated_symptoms": list of strings (nausea, sweating, etc.)
- "severity_descriptors": list of strings (severe, mild, etc.)
- "risk_factors": list of strings (smoking, diabetes, hypertension, etc.)

Do NOT add symptoms not mentioned.
Do NOT guess age or sex.
Do NOT wrap in markdown.

Input Text:
{raw_text}
"""


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


def _extract_text(content) -> str:
    """Extract text from LangChain response content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def extract_symptoms(raw_text: str) -> Dict[str, Any]:
    """Extract structured symptoms from free text."""
    logger.info(f"Extracting symptoms from: {raw_text[:80]}")

    prompt = SYMPTOM_EXTRACTION_PROMPT.format(raw_text=raw_text)

    try:
        llm = get_llm()
        response = llm.invoke([HumanMessage(content=prompt)])
        raw_content = _extract_text(response.content).strip()

        # Remove code fences if present
        if raw_content.startswith("```"):
            raw_content = raw_content.strip("`")
            if raw_content.lower().startswith("json"):
                raw_content = raw_content[4:].lstrip()

        result = json.loads(raw_content)

        # Ensure all keys present
        result.setdefault("age", None)
        result.setdefault("sex", None)
        result.setdefault("chief_complaint", None)
        result.setdefault("symptom_list", [])
        result.setdefault("duration", None)
        result.setdefault("radiation", None)
        result.setdefault("associated_symptoms", [])
        result.setdefault("severity_descriptors", [])
        result.setdefault("risk_factors", [])

        logger.info(f"Extracted: {result.get('symptom_list')}")
        return result

    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)

        err_str = str(e).lower()
        if "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
            error_msg = "AI service rate limit reached. Please try again shortly."
        elif "401" in err_str or "unauthorized" in err_str:
            error_msg = "AI service authentication failed. Check GOOGLE_API_KEY."
        elif "timeout" in err_str or "deadline" in err_str or "504" in err_str:
            error_msg = "AI service timed out. Please try again."
        else:
            error_msg = "Symptom extraction unavailable."

        return {
            "age": None,
            "sex": None,
            "chief_complaint": None,
            "symptom_list": [],
            "duration": None,
            "radiation": None,
            "associated_symptoms": [],
            "severity_descriptors": [],
            "risk_factors": [],
            "error": error_msg,
        }