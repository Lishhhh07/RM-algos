import glob
import os
import numpy as np
import pandas as pd

from src.data_loader import prepare_file
from src.feature_engineering import create_sequences


def main():
    files = glob.glob("data/**/*.csv", recursive=True)

    results = []

    algorithms = sorted(set(
        os.path.basename(os.path.dirname(path)).lower()
        for path in files
    ))

    print("\n========== RTT INFLATION DISTRIBUTION ==========")

    for algorithm in algorithms:
        algorithm_files = [
            path for path in files
            if os.path.basename(os.path.dirname(path)).lower() == algorithm
        ]

        labels = []
        skipped = 0

        for path in algorithm_files:
            try:
                df = prepare_file(path)

                if df is None:
                    skipped += 1
                    continue

                _, y = create_sequences(df)

                if len(y):
                    labels.extend(y[:, 1].astype(int).tolist())

            except Exception:
                skipped += 1

        if not labels:
            continue

        labels = np.asarray(labels)
        positive = int(np.sum(labels == 1))
        negative = int(np.sum(labels == 0))
        total = len(labels)

        results.append({
            "algorithm": algorithm,
            "files": len(algorithm_files),
            "skipped": skipped,
            "sequences": total,
            "positive": positive,
            "negative": negative,
            "positive_rate_percent": round(positive / total * 100, 2)
        })

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(
        "positive_rate_percent",
        ascending=False
    )

    print(result_df.to_string(index=False))

    total_positive = result_df["positive"].sum()
    total_negative = result_df["negative"].sum()
    total = total_positive + total_negative

    print("\n========== OVERALL DISTRIBUTION ==========")
    print("Total sequences:", total)
    print("Positive labels:", total_positive)
    print("Negative labels:", total_negative)
    print("Positive rate:", round(total_positive / total * 100, 2), "%")

    os.makedirs("results", exist_ok=True)
    output = "results/rtt_inflation_distribution_v2.csv"
    result_df.to_csv(output, index=False)

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
