from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.config import WINDOW_SIZE, STRIDE, PREDICTION_HORIZON
from src.data_loader import prepare_file


DATA_DIR = Path("data")
RESULTS_DIR = Path("results")


def ratio(a, b):
    return a / max(abs(b), 1e-9)


def extract_features(path):
    df = prepare_file(path)

    if df is None or len(df) < WINDOW_SIZE:
        return None

    df = df.reset_index(drop=True)

    rtt = df["rtt"].to_numpy(dtype=float)
    cwnd = df["cwnd"].to_numpy(dtype=float)
    ack = df["bytes_acked_delta"].to_numpy(dtype=float)
    retx = df["retransmission_ratio"].to_numpy(dtype=float)

    features = {
        "rtt_relative_level": [],
        "rtt_relative_trend": [],
        "rtt_relative_variation": [],
        "cwnd_relative_level": [],
        "cwnd_relative_change": [],
        "ack_increment_mean": [],
        "retransmission_mean": [],
        "retransmission_frequency": [],
    }

    end = len(df) - PREDICTION_HORIZON + 1
    for i in range(WINDOW_SIZE, end, STRIDE):
        start = i - WINDOW_SIZE

        rw = rtt[start:i]
        cw = cwnd[start:i]
        aw = ack[start:i]
        tw = retx[start:i]

        rtt_median = np.median(rw)
        cwnd_median = np.median(cw)

        features["rtt_relative_level"].append(
            ratio(rw[-1], rtt_median)
        )

        features["rtt_relative_trend"].append(
            ratio(rw[-1] - rw[0], rtt_median)
        )

        features["rtt_relative_variation"].append(
            ratio(np.max(rw) - np.min(rw), rtt_median)
        )

        features["cwnd_relative_level"].append(
            ratio(cw[-1], cwnd_median)
        )

        features["cwnd_relative_change"].append(
            ratio(cw[-1] - cw[0], cwnd_median)
        )

        features["ack_increment_mean"].append(
            np.log1p(max(np.mean(aw), 0))
        )

        features["retransmission_mean"].append(
            np.mean(tw)
        )

        features["retransmission_frequency"].append(
            np.mean(tw > 0)
        )

    return {
        name: np.asarray(values, dtype=float)
        for name, values in features.items()
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    files = sorted(DATA_DIR.glob("*/*.csv"))

    algorithm_features = {}
    usable = 0

    print(f"Files found: {len(files)}")
    print("Extracting relative features...")

    for number, path in enumerate(files, start=1):
        try:
            result = extract_features(path)

            if result is None:
                continue

            algorithm = path.parent.name

            algorithm_features.setdefault(algorithm, {})

            for name, values in result.items():
                algorithm_features[algorithm].setdefault(
                    name, []
                )
                algorithm_features[algorithm][name].append(
                    values
                )

            usable += 1

        except Exception as exc:
            print(f"Skipped {path}: {exc}")

        if number % 50 == 0 or number == len(files):
            print(
                f"Processed {number}/{len(files)} "
                f"| usable: {usable}"
            )

    combined = {}

    for algorithm, feature_map in algorithm_features.items():
        combined[algorithm] = {
            name: np.concatenate(arrays)
            for name, arrays in feature_map.items()
        }

    rows = []

    for algorithm, feature_map in combined.items():
        others = [
            name for name in combined if name != algorithm
        ]

        for feature, values in feature_map.items():
            comparison = np.concatenate([
                combined[name][feature] for name in others
            ])

            rows.append({
                "algorithm": algorithm,
                "feature": feature,
                "algorithm_median": np.median(values),
                "other_median": np.median(comparison),
                "ks_statistic": ks_2samp(
                    values, comparison
                ).statistic,
            })

    result_df = pd.DataFrame(rows)

    result_df.to_csv(
        RESULTS_DIR / "relative_feature_shift.csv",
        index=False,
    )

    summary = (
        result_df.groupby("feature")["ks_statistic"]
        .agg(["mean", "median", "max"])
        .sort_values("mean", ascending=False)
    )

    print("\n========== RELATIVE FEATURE SHIFT ==========")
    print(summary.to_string())

    print("\n========== LARGEST ALGORITHM SHIFTS ==========")

    for feature in summary.index:
        print(f"\n{feature}")
        print(
            result_df[
                result_df["feature"] == feature
            ].nlargest(5, "ks_statistic").to_string(index=False)
        )

    print("\nSaved: results/relative_feature_shift.csv")


if __name__ == "__main__":
    main()
