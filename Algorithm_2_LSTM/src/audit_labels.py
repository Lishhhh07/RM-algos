from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.config import (
    WINDOW_SIZE,
    PREDICTION_HORIZON,
    STRIDE,
    RTT_INFLATION_FACTOR,
)
from src.data_loader import prepare_file


DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
ROLLING_MIN_WINDOW = 50

RAW_REQUIRED = [
    "rtt",
    "cwnd",
    "bytes_sent",
    "bytes_acked",
    "bytes_retrans",
]

FEATURE_NAMES = [
    "rtt_absolute",
    "rtt_over_running_min",
    "rtt_over_rolling_min",
    "rtt_relative_slope",
    "cwnd_absolute",
    "cwnd_growth_ratio",
    "bytes_acked_increment_mean",
    "retransmission_ratio_mean",
]


def safe_ratio(numerator, denominator):
    denominator = np.maximum(np.abs(denominator), 1e-9)
    return numerator / denominator


def audit_file(path):
    raw_rows = 0
    missing_columns = []
    raw_missing_values = 0
    nonpositive_rtt = 0

    try:
        raw = pd.read_csv(path, sep=";")
        raw_rows = len(raw)

        missing_columns = [
            col for col in RAW_REQUIRED if col not in raw.columns
        ]

        available = [
            col for col in RAW_REQUIRED if col in raw.columns
        ]

        if available:
            raw_missing_values = int(
                raw[available].isna().sum().sum()
            )

        if "rtt" in raw.columns:
            raw_rtt = pd.to_numeric(
                raw["rtt"], errors="coerce"
            )
            nonpositive_rtt = int((raw_rtt <= 0).sum())

    except Exception as exc:
        return None, {
            "file": str(path),
            "algorithm": path.parent.name,
            "raw_rows": raw_rows,
            "prepared_rows": 0,
            "missing_required_columns": ",".join(missing_columns),
            "raw_missing_values": raw_missing_values,
            "nonpositive_rtt_rows": nonpositive_rtt,
            "status": f"raw_read_error: {exc}",
        }

    try:
        df = prepare_file(path)

        if df is None or len(df) < WINDOW_SIZE + PREDICTION_HORIZON:
            raise ValueError("File skipped or too few prepared rows")

        required = [
            "rtt",
            "cwnd",
            "bytes_acked_delta",
            "retransmission_ratio",
        ]

        missing_prepared = [
            col for col in required if col not in df.columns
        ]

        if missing_prepared:
            raise ValueError(
                f"Missing prepared columns: {missing_prepared}"
            )

        df = df.reset_index(drop=True)

        rtt = df["rtt"].to_numpy(dtype=float)
        cwnd = df["cwnd"].to_numpy(dtype=float)
        ack = df["bytes_acked_delta"].to_numpy(dtype=float)
        retx = df["retransmission_ratio"].to_numpy(dtype=float)

        running_min = np.minimum.accumulate(rtt)

        rolling_min = (
            pd.Series(rtt)
            .rolling(
                window=ROLLING_MIN_WINDOW,
                min_periods=1,
            )
            .min()
            .to_numpy()
        )

        labels = {
            "endpoint": [],
            "any_in_horizon": [],
            "three_of_five": [],
            "running_min_reference": [],
            "rolling_min_reference": [],
        }

        feature_values = {
            name: [] for name in FEATURE_NAMES
        }

        horizon = PREDICTION_HORIZON
        end = len(df) - horizon + 1

        for i in range(WINDOW_SIZE, end, STRIDE):
            current_idx = i - 1
            future = rtt[i:i + horizon]
            current_rtt = rtt[current_idx]

            if len(future) != horizon:
                continue

            above_current = (
                future > RTT_INFLATION_FACTOR * current_rtt
            )

            labels["endpoint"].append(
                int(above_current[-1])
            )
            labels["any_in_horizon"].append(
                int(np.any(above_current))
            )
            labels["three_of_five"].append(
                int(np.sum(above_current) >= 3)
            )

            labels["running_min_reference"].append(
                int(
                    np.max(future)
                    > RTT_INFLATION_FACTOR * running_min[current_idx]
                )
            )

            labels["rolling_min_reference"].append(
                int(
                    np.max(future)
                    > RTT_INFLATION_FACTOR * rolling_min[current_idx]
                )
            )

            window_start = i - WINDOW_SIZE
            past_rtt = rtt[window_start:i]
            past_cwnd = cwnd[window_start:i]
            past_ack = ack[window_start:i]
            past_retx = retx[window_start:i]

            rtt_slope = (
                past_rtt[-1] - past_rtt[0]
            ) / max(WINDOW_SIZE - 1, 1)

            cwnd_growth = safe_ratio(
                past_cwnd[-1],
                past_cwnd[0],
            )

            feature_values["rtt_absolute"].append(
                current_rtt
            )
            feature_values["rtt_over_running_min"].append(
                safe_ratio(
                    current_rtt,
                    running_min[current_idx],
                )
            )
            feature_values["rtt_over_rolling_min"].append(
                safe_ratio(
                    current_rtt,
                    rolling_min[current_idx],
                )
            )
            feature_values["rtt_relative_slope"].append(
                safe_ratio(rtt_slope, current_rtt)
            )
            feature_values["cwnd_absolute"].append(
                past_cwnd[-1]
            )
            feature_values["cwnd_growth_ratio"].append(
                cwnd_growth
            )
            feature_values["bytes_acked_increment_mean"].append(
                np.mean(past_ack)
            )
            feature_values["retransmission_ratio_mean"].append(
                np.mean(past_retx)
            )

        count = len(labels["endpoint"])

        if count == 0:
            raise ValueError("No sequences generated")

        label_rows = []

        for label_name, values in labels.items():
            positives = int(np.sum(values))

            label_rows.append({
                "file": str(path),
                "algorithm": path.parent.name,
                "label_definition": label_name,
                "sequences": count,
                "positives": positives,
                "negatives": count - positives,
                "positive_rate": positives / count,
            })

        feature_arrays = {
            name: np.asarray(values, dtype=float)
            for name, values in feature_values.items()
        }

        hygiene = {
            "file": str(path),
            "algorithm": path.parent.name,
            "raw_rows": raw_rows,
            "prepared_rows": len(df),
            "missing_required_columns": ",".join(missing_columns),
            "raw_missing_values": raw_missing_values,
            "nonpositive_rtt_rows": nonpositive_rtt,
            "status": "processed",
        }

        return {
            "labels": label_rows,
            "features": feature_arrays,
            "sequence_count": count,
        }, hygiene

    except Exception as exc:
        return None, {
            "file": str(path),
            "algorithm": path.parent.name,
            "raw_rows": raw_rows,
            "prepared_rows": 0,
            "missing_required_columns": ",".join(missing_columns),
            "raw_missing_values": raw_missing_values,
            "nonpositive_rtt_rows": nonpositive_rtt,
            "status": f"preparation_error: {exc}",
        }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    files = sorted(DATA_DIR.glob("*/*.csv"))

    if not files:
        raise FileNotFoundError(
            "No CSV files found under data/<algorithm>/"
        )

    print(f"Files found: {len(files)}")
    print("Running label and feature audit...")

    all_label_rows = []
    hygiene_rows = []
    algorithm_features = {}
    processed_files = 0

    for number, path in enumerate(files, start=1):
        result, hygiene = audit_file(path)
        hygiene_rows.append(hygiene)

        if result is not None:
            processed_files += 1
            all_label_rows.extend(result["labels"])

            algorithm = path.parent.name
            algorithm_features.setdefault(
                algorithm, {}
            )

            for name, values in result["features"].items():
                algorithm_features[algorithm].setdefault(
                    name, []
                )
                algorithm_features[algorithm][name].append(
                    values
                )

        if number % 50 == 0 or number == len(files):
            print(
                f"Processed {number}/{len(files)} files "
                f"| usable: {processed_files}"
            )

    label_file_df = pd.DataFrame(all_label_rows)
    hygiene_df = pd.DataFrame(hygiene_rows)

    label_file_df.to_csv(
        RESULTS_DIR / "label_variants_per_file.csv",
        index=False,
    )

    hygiene_df.to_csv(
        RESULTS_DIR / "data_hygiene_audit.csv",
        index=False,
    )

    if label_file_df.empty:
        raise RuntimeError("No usable files were processed")

    label_algorithm_df = (
        label_file_df
        .groupby(
            ["algorithm", "label_definition"],
            as_index=False,
        )
        .agg(
            sequences=("sequences", "sum"),
            positives=("positives", "sum"),
            negatives=("negatives", "sum"),
        )
    )

    label_algorithm_df["positive_rate"] = (
        label_algorithm_df["positives"]
        / label_algorithm_df["sequences"]
    )

    label_algorithm_df.to_csv(
        RESULTS_DIR / "label_variants_per_algorithm.csv",
        index=False,
    )

    combined_features = {}

    for algorithm, feature_map in algorithm_features.items():
        combined_features[algorithm] = {}

        for name, arrays in feature_map.items():
            combined_features[algorithm][name] = np.concatenate(
                arrays
            )

    shift_rows = []

    for algorithm, feature_map in combined_features.items():
        other_algorithms = [
            other for other in combined_features
            if other != algorithm
        ]

        for feature_name, values in feature_map.items():
            comparison = np.concatenate([
                combined_features[other][feature_name]
                for other in other_algorithms
            ])

            ks_statistic = ks_2samp(
                values,
                comparison,
            ).statistic

            shift_rows.append({
                "algorithm": algorithm,
                "feature": feature_name,
                "algorithm_sequences": len(values),
                "other_algorithms_sequences": len(comparison),
                "algorithm_median": float(np.median(values)),
                "other_algorithms_median": float(
                    np.median(comparison)
                ),
                "algorithm_q25": float(
                    np.percentile(values, 25)
                ),
                "algorithm_q75": float(
                    np.percentile(values, 75)
                ),
                "other_q25": float(
                    np.percentile(comparison, 25)
                ),
                "other_q75": float(
                    np.percentile(comparison, 75)
                ),
                "ks_statistic": float(ks_statistic),
            })

    shift_df = pd.DataFrame(shift_rows)

    shift_df.to_csv(
        RESULTS_DIR / "feature_shift_ks.csv",
        index=False,
    )

    print("\n========== AUDIT COMPLETE ==========")
    print(f"Total files: {len(files)}")
    print(f"Usable files: {processed_files}")
    print(
        "Skipped/error files:",
        len(files) - processed_files,
    )

    print("\nLabel prevalence by algorithm:")
    print(
        label_algorithm_df[
            [
                "algorithm",
                "label_definition",
                "sequences",
                "positive_rate",
            ]
        ].to_string(index=False)
    )

    print("\nHygiene status:")
    print(hygiene_df["status"].value_counts().to_string())

    print("\nSaved:")
    print("results/label_variants_per_file.csv")
    print("results/label_variants_per_algorithm.csv")
    print("results/feature_shift_ks.csv")
    print("results/data_hygiene_audit.csv")


if __name__ == "__main__":
    main()
