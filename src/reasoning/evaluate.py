import os
import sys
import json
import logging
import warnings

# Suppress standard Python logging and warnings
logging.getLogger("google.genai").setLevel(logging.ERROR)
logging.getLogger("google.genai.models").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

class SuppressStderr:
    """Context manager to suppress direct stderr output from external C-extensions and SDKs."""
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self._original_stderr

with SuppressStderr():
    from src.reasoning.reasoner import reason


def load_test_cases(data_path: str = "tests/data/eval_cases.json") -> list:
    """Loads test cases from a local JSON dataset if available, otherwise falls back to sample cases."""
    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    return [
        {
            "id": "CASE-001",
            "symptoms": {"symptom_list": ["crushing chest pain", "diaphoresis"], "age": 58, "sex": "male"},
            "ecg_findings": {"class": "STEMI", "confidence": 0.96},
            "evidence": [
                {"chunk": "ST-segment elevation myocardial infarction requires immediate reperfusion therapy.", "source": "acc_aha_guidelines"}
            ],
            "expected_severity": "Critical"
        },
        {
            "id": "CASE-002",
            "symptoms": {"symptom_list": ["palpitations", "mild fatigue"], "age": 34, "sex": "female"},
            "ecg_findings": {"class": "NORM", "confidence": 0.92},
            "evidence": [
                {"chunk": "Normal sinus rhythm with no acute ischemic changes.", "source": "triage_protocol"}
            ],
            "expected_severity": "Non-Urgent"
        }
    ]


def run_evaluation(
    test_cases: list,
    output_path: str = "docs/evaluation/llm_reasoning_metrics.json"
):
    """Executes reasoning evaluation over test cases and writes benchmark metrics to file."""
    results = []
    successful_parses = 0

    for case in test_cases:
        # Suppress stderr during model invocation to catch SDK runtime warnings
        with SuppressStderr():
            res = reason(case["symptoms"], case["ecg_findings"], case["evidence"])
        
        is_parsed = res.get("severity") != "Unknown" and bool(res.get("explanation"))
        if is_parsed:
            successful_parses += 1

        results.append({
            "case_id": case.get("id"),
            "expected_severity": case.get("expected_severity"),
            "predicted_severity": res.get("severity"),
            "conditions": res.get("conditions", []),
            "citations": res.get("citations", []),
            "output": res
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    metrics = {
        "total_cases_evaluated": len(results),
        "structured_json_parse_rate": round(successful_parses / max(len(results), 1), 2),
        "results": results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Evaluation metrics successfully written to {output_path}")


if __name__ == "__main__":
    cases = load_test_cases()
    run_evaluation(cases)