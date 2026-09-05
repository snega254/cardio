"""
data_loader.py

Responsible for:
  1. Loading the PTB-XL metadata (ptbxl_database.csv, scp_statements.csv)
  2. Parsing the `scp_codes` column (stored as a stringified dict) into a
     single-label mapping onto PTB-XL's 5 diagnostic superclasses:
     NORM, MI, STTC, CD, HYP
  3. Loading the raw waveform signals via the `wfdb` package.

PTB-XL layout expected (standard download from PhysioNet):

  data/raw/ptbxl/
    ├── ptbxl_database.csv
    ├── scp_statements.csv
    ├── records100/          # 100 Hz signals
    └── records500/          # 500 Hz signals

Nothing in this file writes anywhere outside data/raw (gitignored) —
it only reads.
"""

import ast
import os

import numpy as np
import pandas as pd
import wfdb

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def load_metadata(ptbxl_root: str) -> pd.DataFrame:
    """
    Load ptbxl_database.csv and attach a `scp_codes` dict column.

    Args:
        ptbxl_root: path to the folder containing ptbxl_database.csv

    Returns:
        DataFrame indexed by ecg_id, with scp_codes parsed to dict.
    """
    csv_path = os.path.join(ptbxl_root, "ptbxl_database.csv")
    df = pd.read_csv(csv_path, index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    return df


def load_scp_statements(ptbxl_root: str) -> pd.DataFrame:
    """
    Load scp_statements.csv, which maps individual SCP-ECG statement
    codes (e.g. 'IMI', 'NDT') to one of the 5 diagnostic superclasses.
    """
    csv_path = os.path.join(ptbxl_root, "scp_statements.csv")
    agg_df = pd.read_csv(csv_path, index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]
    return agg_df


def aggregate_diagnostic(scp_dict: dict, agg_df: pd.DataFrame) -> list:
    """
    Map one record's scp_codes dict -> list of superclasses it belongs to.

    A record can technically map to multiple superclasses; for a clean
    single-label problem we resolve to the superclass with the highest
    confidence code, with a fixed tie-break order matching SUPERCLASSES.
    """
    hits = {}
    for code, confidence in scp_dict.items():
        if code in agg_df.index:
            superclass = agg_df.loc[code, "diagnostic_class"]
            if superclass in SUPERCLASSES:
                hits[superclass] = max(hits.get(superclass, 0), confidence)
    return list(hits.keys())


def add_superclass_labels(df: pd.DataFrame, agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add two columns to df:
      - 'superclasses': list of all applicable superclasses (for reference)
      - 'label': single resolved label (str), used for training

    Records with zero matching superclasses are dropped (rare, but PTB-XL
    has a handful of records with only non-diagnostic codes).
    """
    df = df.copy()
    df["superclasses"] = df["scp_codes"].apply(
        lambda d: aggregate_diagnostic(d, agg_df)
    )
    df = df[df["superclasses"].map(len) > 0].reset_index()

    def resolve_single_label(classes):
        for sc in SUPERCLASSES:  # fixed priority order
            if sc in classes:
                return sc
        return classes[0]

    df["label"] = df["superclasses"].apply(resolve_single_label)
    df = df.set_index("ecg_id")
    return df


def load_raw_signal(record_path: str, sampling_rate: int = 100) -> np.ndarray:
    """
    Load a single record's raw 12-lead signal via wfdb.

    Args:
        record_path: path to the record WITHOUT extension, e.g.
            'data/raw/ptbxl/records100/00000/00001_lr'
        sampling_rate: 100 or 500, must match the folder used.

    Returns:
        np.ndarray of shape (n_samples, 12) — raw, unfiltered.
    """
    record = wfdb.rdrecord(record_path)
    return record.p_signal.astype(np.float32)


def load_signals_for_split(
    df: pd.DataFrame, ptbxl_root: str, sampling_rate: int = 100
) -> np.ndarray:
    """
    Bulk-load signals for every row in df, using the correct
    filename_lr / filename_hr column depending on sampling_rate.

    Returns:
        np.ndarray of shape (n_records, n_samples, 12)
    """
    filename_col = "filename_lr" if sampling_rate == 100 else "filename_hr"
    signals = []
    for fname in df[filename_col]:
        record_path = os.path.join(ptbxl_root, fname)
        signals.append(load_raw_signal(record_path, sampling_rate))
    return np.stack(signals, axis=0)


if __name__ == "__main__":
    # quick manual smoke test — not part of the automated test suite
    root = os.environ.get("PTBXL_ROOT", "data/raw/ptbxl")
    meta = load_metadata(root)
    agg = load_scp_statements(root)
    labeled = add_superclass_labels(meta, agg)
    print(labeled["label"].value_counts())
