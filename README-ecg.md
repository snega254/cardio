# Member 1 — ECG Model (`src/ecg/`)

## Setup

```bash
pip install -r requirements-ecg.txt
```

Download PTB-XL from PhysioNet and place it (or symlink it) at
`data/raw/ptbxl/`, containing `ptbxl_database.csv`, `scp_statements.csv`,
`records100/`, `records500/`. Set the env var if you keep it elsewhere:

```bash
export PTBXL_ROOT=/path/to/ptbxl
```

## Run the tests (no dataset needed — synthetic data only)

```bash
pytest tests/ecg/ -v
```

## Train a model

```bash
python -m src.ecg.train --model cnn1d --epochs 30
python -m src.ecg.train --model resnet1d --epochs 30
python -m src.ecg.train --model inception1d --epochs 30
```

Each run saves its best checkpoint to `checkpoints/<model>_best.pt`.

## Evaluate all trained models and write metrics

```bash
python -m src.ecg.evaluate
```

Writes `docs/evaluation/ecg_model_metrics.json` (per-class
Precision/Recall/F1, macro-F1, macro-AUROC, AUPRC, confusion matrix)
for every model that has a checkpoint.

## The interface the rest of the team uses

```python
from src.ecg.interface import predict

predicted_class, confidence, gradcam_map = predict(raw_12_lead_signal)
# predicted_class: one of "NORM", "MI", "STTC", "CD", "HYP"
# confidence: float in [0, 1]
# gradcam_map: np.ndarray, same length as the signal, importance over time
```

## File map

| File | Purpose |
|---|---|
| `data_loader.py` | Load PTB-XL metadata + signals, map scp_codes → 5 superclasses |
| `preprocessing.py` | Bandpass filter, 12→8 lead reduction, normalization |
| `dataset.py` | PyTorch `Dataset` using official `strat_fold` split |
| `models/cnn1d.py` | Primary custom 1D CNN |
| `models/resnet1d.py` | Published baseline #1 |
| `models/inception1d.py` | Published baseline #2 |
| `models/feature_baseline.py` | Optional RF/XGBoost baseline |
| `train.py` | Trains any of the 3 deep models |
| `gradcam.py` | Grad-CAM explainability (CNN1D only) |
| `evaluate.py` | Scores all trained models, writes metrics JSON |
| `interface.py` | **`predict()` — the contract `src/pipeline.py` calls** |
