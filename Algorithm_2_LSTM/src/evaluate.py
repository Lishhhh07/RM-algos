import json
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, average_precision_score,
    mean_absolute_error, mean_squared_error, r2_score
)

from .config import METRICS_PATH_V2, PREDICTIONS_PATH_V2


def evaluate_model(model, X_test, y_test, target_scaler):
    predictions = model.predict(X_test, verbose=0)

    rtt_pred_scaled = np.asarray(
        predictions["rtt_delta"]
    ).reshape(-1, 1)

    rtt_pred = target_scaler.inverse_transform(
        rtt_pred_scaled
    ).ravel()

    inflation_prob = np.asarray(
        predictions["rtt_inflation"]
    ).ravel()

    retrans_prob = np.asarray(
        predictions["retransmission"]
    ).ravel()

    inflation_true = y_test[:, 1].astype(int)
    retrans_true = y_test[:, 2].astype(int)

    inflation_pred = (inflation_prob >= 0.5).astype(int)
    retrans_pred = (retrans_prob >= 0.5).astype(int)

    results = {
        "RTT Delta Regression": {
            "MAE": float(mean_absolute_error(y_test[:, 0], rtt_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test[:, 0], rtt_pred))),
            "R2": float(r2_score(y_test[:, 0], rtt_pred))
        }
    }

    for name, actual, predicted, probabilities in [
        ("RTT Inflation", inflation_true, inflation_pred, inflation_prob),
        ("Retransmission Risk", retrans_true, retrans_pred, retrans_prob)
    ]:
        tn, fp, fn, tp = confusion_matrix(
            actual, predicted, labels=[0, 1]
        ).ravel()

        try:
            pr_auc = float(
                average_precision_score(actual, probabilities)
            )
        except ValueError:
            pr_auc = None

        results[name] = {
            "accuracy": float(accuracy_score(actual, predicted)),
            "precision": float(
                precision_score(actual, predicted, zero_division=0)
            ),
            "recall": float(
                recall_score(actual, predicted, zero_division=0)
            ),
            "f1": float(f1_score(actual, predicted, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
            "pr_auc": pr_auc,
            "positive_rate": float(np.mean(actual)),
            "confusion_matrix": {
                "TN": int(tn), "FP": int(fp),
                "FN": int(fn), "TP": int(tp)
            }
        }

    Path("results").mkdir(exist_ok=True)

    with open(METRICS_PATH_V2, "w") as f:
        json.dump(results, f, indent=4)

    pd.DataFrame({
        "actual_rtt_delta": y_test[:, 0],
        "predicted_rtt_delta": rtt_pred,
        "actual_rtt_inflation": inflation_true,
        "predicted_rtt_inflation_probability": inflation_prob,
        "actual_retransmission": retrans_true,
        "predicted_retransmission_probability": retrans_prob
    }).to_csv(PREDICTIONS_PATH_V2, index=False)

    print("\n===== V2 TEST RESULTS =====")

    for task, metrics in results.items():
        print("\n", task)
        for metric, value in metrics.items():
            print(f"{metric}: {value}")

    print("\nMetrics saved:", METRICS_PATH_V2)
    print("Predictions saved:", PREDICTIONS_PATH_V2)

    return results
