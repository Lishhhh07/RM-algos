import json
import time
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score,
    f1_score, average_precision_score
)

from .data_loader import load_dataset
from .config import RANDOM_SEED


def flatten_sequences(X):
    return X.reshape(len(X), -1)


def regression_metrics(y_true, predictions):
    return {
        "MAE": float(mean_absolute_error(y_true, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "R2": float(r2_score(y_true, predictions))
    }


def classification_metrics(y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(
            y_true, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_true, predictions, zero_division=0
        )),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probabilities))
    }


def main():
    start = time.time()

    print("\n===== STEP 1: LOAD DATA =====")
    (X_train, y_train), (_, _), (X_test, y_test) = load_dataset()

    X_train = flatten_sequences(X_train)
    X_test = flatten_sequences(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Input features:", X_train.shape[1])

    results = {}

    print("\n===== STEP 2: RTT DELTA REGRESSION =====")

    regressors = {
        "Dummy Regressor": DummyRegressor(strategy="mean"),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100,
            max_depth=18,
            min_samples_leaf=2,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
    }

    for name, model in regressors.items():
        print("\nTraining:", name)
        model.fit(X_train, y_train[:, 0])
        predictions = model.predict(X_test)
        results[name + " - RTT Delta"] = regression_metrics(
            y_test[:, 0], predictions
        )
        print(results[name + " - RTT Delta"])

    print("\n===== STEP 3: CLASSIFICATION TASKS =====")

    tasks = {
        "RTT Inflation": 1,
        "Retransmission Risk": 2
    }

    classifiers = {
        "Dummy Classifier": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(
            max_iter=500,
            random_state=RANDOM_SEED
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=100,
            max_depth=18,
            min_samples_leaf=2,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
    }

    for task_name, target_index in tasks.items():
        y_train_task = y_train[:, target_index].astype(int)
        y_test_task = y_test[:, target_index].astype(int)

        print("\n---", task_name, "---")

        for model_name, model in classifiers.items():
            print("\nTraining:", model_name)
            model.fit(X_train, y_train_task)

            probabilities = model.predict_proba(X_test)[:, 1]

            key = model_name + " - " + task_name
            results[key] = classification_metrics(
                y_test_task, probabilities
            )
            print(results[key])

    print("\n===== STEP 4: ADD EXISTING LSTM RESULTS =====")

    with open("results/evaluation_metrics.json") as f:
        lstm = json.load(f)

    results["LSTM - RTT Delta"] = lstm["RTT Delta Regression"]

    for task_name in tasks:
        results["LSTM - " + task_name] = lstm[task_name]

    Path("results").mkdir(exist_ok=True)

    with open("results/baseline_comparison.json", "w") as f:
        json.dump(results, f, indent=4)

    rows = []
    for model_name, metrics in results.items():
        row = {"Model": model_name}
        row.update(metrics)
        rows.append(row)

    pd.DataFrame(rows).to_csv(
        "results/baseline_comparison.csv", index=False
    )

    print("\n===== FINAL COMPARISON =====")
    print(pd.DataFrame(rows).to_string(index=False))

    elapsed = (time.time() - start) / 60

    print("\nSaved: results/baseline_comparison.json")
    print("Saved: results/baseline_comparison.csv")
    print(f"Execution time: {elapsed:.2f} minutes")
    print("BASELINE COMPARISON COMPLETED.")


if __name__ == "__main__":
    main()
