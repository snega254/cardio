"""
feature_baseline.py

Optional simple baseline: hand-crafted statistical features per lead
fed into Random Forest / XGBoost. Useful as a sanity floor to confirm
the deep models are actually adding value.

Not part of the required deliverables — skip if short on time.
"""

from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def extract_features(signal: np.ndarray) -> np.ndarray:
    """
    Extract simple per-lead statistical features from a preprocessed
    (n_samples, n_leads) signal: mean, std, min, max, and mean
    absolute first difference (a crude slope/variability measure).

    Returns:
        1D feature vector of length n_leads * 5.
    """
    feats = []
    for lead in range(signal.shape[1]):
        x = signal[:, lead]
        diff = np.abs(np.diff(x))
        feats.extend([x.mean(), x.std(), x.min(), x.max(), diff.mean()])
    return np.array(feats, dtype=np.float32)


def extract_features_batch(signals: np.ndarray) -> np.ndarray:
    """signals: (n, n_samples, n_leads) -> (n, n_leads*5)"""
    return np.stack([extract_features(s) for s in signals], axis=0)


def build_random_forest(random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300, max_depth=None, random_state=random_state, n_jobs=-1
    )


def build_xgboost(random_state: int = 42) -> Optional["XGBClassifier"]:
    if not HAS_XGBOOST:
        return None
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=random_state,
        eval_metric="mlogloss",
    )
