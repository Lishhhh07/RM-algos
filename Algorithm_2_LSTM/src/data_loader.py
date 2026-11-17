import glob
import os
import numpy as np
import pandas as pd

from .config import (
    RANDOM_SEED,
    TRAIN_RATIO,
    VALIDATION_RATIO,
    MIN_FLOW_ROWS
)
from .feature_engineering import create_sequences


def split_files(files):
    """
    Deterministic random file-level split.
    Different files from the same algorithm may appear in different partitions.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    files = sorted(files)
    rng.shuffle(files)

    n = len(files)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VALIDATION_RATIO)

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    print()
    print("===== RANDOM FILE-LEVEL SPLIT =====")
    print("Train files:", len(train_files))
    print("Validation files:", len(val_files))
    print("Test files:", len(test_files))
    print("Algorithm overlap is allowed.")

    assert not (set(train_files) & set(val_files))
    assert not (set(train_files) & set(test_files))
    assert not (set(val_files) & set(test_files))

    return train_files, val_files, test_files

def prepare_file(path):
    df = pd.read_csv(path, sep=";")
    df = df.dropna(axis=1, how="all")

    required = [
        "rtt", "cwnd", "bytes_sent",
        "bytes_acked", "bytes_retrans"
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required).reset_index(drop=True)

    if len(df) < MIN_FLOW_ROWS:
        return None

    # Convert cumulative counters into per-sample increments.
    # If a counter decreases, treat it as a reset and use its new value.
    for col in ["bytes_sent", "bytes_acked", "bytes_retrans"]:
        delta = df[col].diff()
        delta.iloc[0] = 0

        reset_mask = delta < 0
        delta.loc[reset_mask] = df.loc[reset_mask, col]

        df[col + "_delta"] = delta.clip(lower=0)

    denominator = df["bytes_sent_delta"].clip(lower=1)
    df["retransmission_ratio"] = (
        df["bytes_retrans_delta"] / denominator
    )

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[
        "rtt", "cwnd", "bytes_acked_delta",
        "retransmission_ratio"
    ]).reset_index(drop=True)

    return df


def build_partition(paths, name):
    all_X, all_y = [], []
    skipped = 0

    for index, path in enumerate(paths, start=1):
        try:
            df = prepare_file(path)
            if df is None:
                skipped += 1
                continue

            X, y = create_sequences(df)
            if len(X):
                all_X.append(X)
                all_y.append(y)

        except Exception as exc:
            skipped += 1
            print(f"Skipped {path}: {exc}")

        if index % 50 == 0 or index == len(paths):
            print(f"{name}: processed {index}/{len(paths)} files")

    if not all_X:
        raise ValueError(f"No sequences generated for {name}.")

    X = np.concatenate(all_X).astype(np.float32)
    y = np.concatenate(all_y).astype(np.float32)

    print(
        f"{name}: {len(paths)} files, "
        f"{len(X):,} sequences, {skipped} skipped"
    )
    return X, y


def load_dataset(data_dir="data"):
    files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)

    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    train_files, val_files, test_files = split_files(files)

    print("Files:", len(files))
    print(
        "Split:",
        len(train_files), "train |",
        len(val_files), "validation |",
        len(test_files), "test"
    )

    train = build_partition(train_files, "TRAIN")
    val = build_partition(val_files, "VALIDATION")
    test = build_partition(test_files, "TEST")

    return train, val, test
