"""
Verify Members 1, 2, 3 are all working correctly.
"""
import sys
from pathlib import Path
sys.path.insert(0, '.')

print("=" * 70)
print("CARDIOMAGENT - MEMBER VERIFICATION")
print("=" * 70)

# ---------------- MEMBER 1 ----------------
print("\n[MEMBER 1] ECG Model")
print("-" * 70)

try:
    import torch
    import wfdb
    from src.ecg.interface import predict

    # Model files
    for m in ['inception1d_best.pt', 'cnn1d_best.pt', 'resnet1d_best.pt']:
        p = Path('checkpoints') / m
        status = "OK" if p.exists() else "MISSING"
        print(f"  {status}: {m}")

    # Real predictions
    print("\n  Testing on real PTB-XL records:")
    for rec_id in ['00001_lr', '00003_lr', '00005_lr']:
        rec = wfdb.rdrecord(f'data/raw/ptbxl/records100/00000/{rec_id}')
        cls, conf, _ = predict(rec.p_signal, return_gradcam=False)
        print(f"    {rec_id}: {cls} ({conf:.4f})")

    print("\n  MEMBER 1: OK")

except Exception as e:
    print(f"  MEMBER 1 FAILED: {e}")

# ---------------- MEMBER 2 ----------------
print("\n[MEMBER 2] RAG")
print("-" * 70)

try:
    from src.rag.retriever import retrieve

    for q in ['chest pain', 'atrial fibrillation']:
        results = retrieve(q, top_k=2)
        print(f"  Query '{q}': {len(results)} results")
        for r in results[:1]:
            print(f"    [{r['score']:.3f}] {r['source']}")

    # Same meaning test
    r1 = retrieve('heart attack', top_k=1)
    r2 = retrieve('myocardial infarction', top_k=1)
    print(f"  'heart attack' -> {r1[0]['source']}")
    print(f"  'myocardial infarction' -> {r2[0]['source']}")

    print("\n  MEMBER 2: OK")

except Exception as e:
    print(f"  MEMBER 2 FAILED: {e}")

# ---------------- MEMBER 3 ----------------
print("\n[MEMBER 3] LLM Reasoning")
print("-" * 70)

try:
    from src.reasoning.reasoner import reason, get_llm
    from src.reasoning.symptom_extractor import extract_symptoms

    # Test symptom extraction
    sym = extract_symptoms('55 year old male with chest pain for 2 hours')
    print(f"  Symptom extraction: {sym.get('symptom_list', [])}")

    # Test RAG influence
    symptoms = {
        'age': 55, 'sex': 'Male', 'chief_complaint': 'chest pain',
        'symptom_list': ['chest pain'], 'duration': '2h',
        'severity_descriptors': [], 'radiation': None,
        'associated_symptoms': [], 'risk_factors': []
    }
    urgent = [{'chunk': 'Suspected ACS requires immediate emergency evaluation.', 'source': 'urgent.txt', 'score': 0.95}]
    mild = [{'chunk': 'Atypical chest pain can be managed outpatient.', 'source': 'mild.txt', 'score': 0.95}]

    r1 = reason(symptoms, {}, urgent)
    r2 = reason(symptoms, {}, mild)

    print(f"  Urgent evidence -> {r1.get('severity')}")
    print(f"  Mild evidence -> {r2.get('severity')}")

    if r1.get('severity') != r2.get('severity'):
        print("  RAG influence: CONFIRMED")
    else:
        print("  RAG influence: WEAK")

    print("\n  MEMBER 3: OK")

except Exception as e:
    print(f"  MEMBER 3 FAILED: {e}")

# ---------------- INTEGRATION ----------------
print("\n[INTEGRATION] Full Pipeline")
print("-" * 70)

try:
    from src.pipeline import run_pipeline_safe

    result = run_pipeline_safe('55 year old male with chest pain radiating to left arm')
    print(f"  Status: {result.get('pipeline_status')}")
    print(f"  Severity: {result.get('severity')}")
    print(f"  Conditions: {len(result.get('conditions', []))}")
    print(f"  ECG class: {result.get('ecg_analysis', {}).get('predicted_class')}")

    print("\n  INTEGRATION: OK")

except Exception as e:
    print(f"  INTEGRATION FAILED: {e}")

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)