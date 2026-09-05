import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        response_mime_type="application/json"
    )

REASONING_PROMPT = """
You are CardioAgent, an advanced decision-support assistant for general physicians.
Synthesize the provided patient symptoms, ECG classification findings, and retrieved clinical guidelines to generate a structured decision-support output.

Output MUST be a valid JSON object with:
- "severity": string ("Critical", "Urgent", or "Non-Urgent")
- "conditions": list of strings (potential diagnoses or findings)
- "explanation": string (clinical narrative explaining the reasoning)
- "citations": list of strings (sources cited from evidence)
- "limitations": string

Symptoms:
{symptoms}

ECG Findings:
{ecg_findings}

Retrieved Evidence:
{evidence}
"""

def reason(symptoms: dict, ecg_findings: dict, evidence: list) -> dict:
    formatted_prompt = REASONING_PROMPT.format(
        symptoms=json.dumps(symptoms),
        ecg_findings=json.dumps(ecg_findings),
        evidence=json.dumps(evidence)
    )
    
    llm = get_llm()
    response = llm.invoke([HumanMessage(content=formatted_prompt)])
    
    content = response.content
    if not isinstance(content, str):
        content = str(content)

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {
            "severity": "Unknown",
            "conditions": [],
            "explanation": content,
            "citations": [],
            "limitations": "Failed to parse structured JSON response from model."
        }