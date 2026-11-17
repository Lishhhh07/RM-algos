import json
import numpy as np
import tensorflow as tf
import joblib

from pathlib import Path
from sklearn.metrics import (
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

from .data_loader import load_dataset
from .config import MODEL_PATH_V2, SCALER_PATH_V2


def scale_data(X, scaler):
    return scaler.transform(
        X.reshape(-1, X.shape[-1])
    ).reshape(X.shape).astype(np.float32)


def get_best_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )

    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-10)
    )

    index = int(np.argmax(f1_values))

    return float(thresholds[index]), float(f1_values[index])


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),
        "predicted_positive_rate": float(np.mean(predictions))
    }


def main():
    print("Loading dataset...")
    (_, _), (X_val, y_val), (X_test, y_test) = load_dataset()

    print("\nLoading V2 model and scaler...")
    model = tf.keras.models.load_model(MODEL_PATH_V2)
    scaler = joblib.load(SCALER_PATH_V2)

    X_val_scaled = scale_data(X_val, scaler)
    X_test_scaled = scale_data(X_test, scaler)

    print("\nGenerating validation predictions...")
    val_predictions = model.predict(X_val_scaled, verbose=0)

    print("Generating test predictions...")
    test_predictions = model.predict(X_test_scaled, verbose=0)

    tasks = {
        "RTT Inflation": {
            "column": 1,
            "output": "rtt_inflation"
        },
        "Retransmission Risk": {
            "column": 2,
            "output": "retransmission"
        }
    }

    results = {}

    for name, details in tasks.items():
        column = details["column"]
        output = details["output"]

        val_true = y_val[:, column].astype(int)
        test_true = y_test[:, column].astype(int)

        val_prob = np.asarray(
            val_predictions[output]
        ).ravel()

        test_prob = np.asarray(
            test_predictions[output]
        ).ravel()

        threshold, val_best_f1 = get_best_threshold(
            val_true, val_prob
        )

        results[name] = {
            "validation_positive_rate": float(np.mean(val_true)),
            "validation_best_f1": val_best_f1,
            "validation_selected_threshold": threshold,
            "test_default_threshold_0.5": calculate_metrics(
                test_true, test_prob, 0.5
            ),
            "test_validation_selected_threshold": calculate_metrics(
                test_true, test_prob, threshold
            )
        }

    Path("results").mkdir(exist_ok=True)

    output_path = "results/threshold_analysis_v2.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print("\n========== THRESHOLD ANALYSIS ==========")

    for name, values in results.items():
        print("\n", name)
        print("Validation positive rate:",
              values["validation_positive_rate"])
        print("Selected threshold:",
              values["validation_selected_threshold"])
        print("Validation best F1:",
              values["validation_best_f1"])
        print("Test at 0.5:",
              values["test_default_threshold_0.5"])
        print("Test at selected threshold:",
              values["test_validation_selected_threshold"])

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
