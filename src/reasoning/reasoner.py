import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from src.reasoning.prompts import REASONING_PROMPT

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        response_mime_type="application/json"
    )

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