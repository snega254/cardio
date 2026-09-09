import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        response_mime_type="application/json"
    )

SYMPTOM_EXTRACTION_PROMPT = """
You are a clinical AI assistant. Extract structured patient symptom information from the raw clinical input below.
Return a valid JSON object with the following keys:
- "age": integer or null
- "sex": string or null
- "symptom_list": list of strings
- "duration": string or null
- "severity_descriptors": list of strings

Input Text:
{raw_text}
"""

def extract_symptoms(raw_text: str) -> dict:
    prompt = SYMPTOM_EXTRACTION_PROMPT.format(raw_text=raw_text)
    llm = get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    
    content = response.content
    if not isinstance(content, str):
        content = str(content)

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"symptom_list": [], "raw_output": content}