import pytest
from unittest.mock import patch, MagicMock
from src.reasoning.symptom_extractor import extract_symptoms
from src.reasoning.reasoner import reason

@patch("src.reasoning.symptom_extractor.get_llm")
def test_extract_symptoms(mock_get_llm):
    mock_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.content = '{"age": 55, "sex": "male", "symptom_list": ["sharp chest pain"], "duration": "2 hours", "severity_descriptors": ["sharp"]}'
    mock_instance.invoke.return_value = mock_response
    mock_get_llm.return_value = mock_instance

    sample_text = "55yo male experiencing sharp chest pain for 2 hours"
    res = extract_symptoms(sample_text)
    
    assert isinstance(res, dict)
    assert "symptom_list" in res
    assert res["symptom_list"] == ["sharp chest pain"]


@patch("src.reasoning.reasoner.get_llm")
def test_reason_interface(mock_get_llm):
    mock_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.content = '{"severity": "Non-Urgent", "conditions": ["Normal"], "explanation": "ECG is within normal limits.", "citations": ["triage_ref"], "limitations": "None"}'
    mock_instance.invoke.return_value = mock_response
    mock_get_llm.return_value = mock_instance

    symptoms = {"symptom_list": ["chest pain"]}
    ecg_findings = {"class": "NORM", "confidence": 0.98}
    evidence = [{"chunk": "Normal ECG rule-out protocol.", "source": "triage_ref"}]
    
    output = reason(symptoms, ecg_findings, evidence)
    
    assert "severity" in output
    assert "conditions" in output
    assert "explanation" in output
    assert "citations" in output
    assert "limitations" in output