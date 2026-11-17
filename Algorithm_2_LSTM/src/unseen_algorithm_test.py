import os
import glob
import json
import time
import joblib
import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score,
    f1_score, average_precision_score
)

from .config import (
    RANDOM_SEED, EPOCHS, BATCH_SIZE,
    EARLY_STOPPING_PATIENCE
)
from .data_loader import build_partition
from .model import build_lstm_model
from .train import make_targets


def split_seen_algorithms(files, held_out):
    rng = np.random.default_rng(RANDOM_SEED)
    grouped = {}

    for path in files:
        algorithm = os.path.basename(os.path.dirname(path))
        if algorithm not in held_out:
            grouped.setdefault(algorithm, []).append(path)

    train_files = []
    val_files = []

    for algorithm in sorted(grouped):
        group = np.array(sorted(grouped[algorithm]), dtype=object)
        rng.shuffle(group)

        cut = int(len(group) * 0.85)
        train_files.extend(group[:cut].tolist())
        val_files.extend(group[cut:].tolist())

    return train_files, val_files


def scale_data(X_train, X_val, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train.reshape(-1, X_train.shape[-1])
    ).reshape(X_train.shape).astype(np.float32)

    X_val_scaled = scaler.transform(
        X_val.reshape(-1, X_val.shape[-1])
    ).reshape(X_val.shape).astype(np.float32)

    X_test_scaled = scaler.transform(
        X_test.reshape(-1, X_test.shape[-1])
    ).reshape(X_test.shape).astype(np.float32)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def evaluate(y_true, predictions):
    rtt = np.asarray(predictions["rtt_delta"]).ravel()
    inflation = np.asarray(predictions["rtt_inflation"]).ravel()
    retrans = np.asarray(predictions["retransmission"]).ravel()

    output = {
        "RTT Delta": {
            "MAE": float(mean_absolute_error(y_true[:, 0], rtt)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true[:, 0], rtt))),
            "R2": float(r2_score(y_true[:, 0], rtt))
        }
    }

    for name, index, probabilities in [
        ("RTT Inflation", 1, inflation),
        ("Retransmission Risk", 2, retrans)
    ]:
        actual = y_true[:, index].astype(int)
        predicted = (probabilities >= 0.5).astype(int)

        output[name] = {
            "accuracy": float(accuracy_score(actual, predicted)),
            "precision": float(precision_score(
                actual, predicted, zero_division=0
            )),
            "recall": float(recall_score(
                actual, predicted, zero_division=0
            )),
            "f1": float(f1_score(
                actual, predicted, zero_division=0
            )),
            "pr_auc": float(average_precision_score(
                actual, probabilities
            ))
        }

    return output


def main():
    start = time.time()
    held_out = {"bbr", "cubic", "reno"}

    print("\n===== STEP 1: COLLECT FILES =====")
    files = glob.glob("data/**/*.csv", recursive=True)

    available = {
        os.path.basename(os.path.dirname(path))
        for path in files
    }

    missing = held_out - available
    if missing:
        raise ValueError(f"Missing held-out algorithms: {missing}")

    test_files = [
        path for path in files
        if os.path.basename(os.path.dirname(path)) in held_out
    ]

    train_files, val_files = split_seen_algorithms(files, held_out)

    print("Training files:", len(train_files))
    print("Validation files:", len(val_files))
    print("Unseen test algorithms:", sorted(held_out))
    print("Unseen test files:", len(test_files))

    print("\n===== STEP 2: BUILD DATA PARTITIONS =====")
    X_train, y_train = build_partition(train_files, "TRAIN")
    X_val, y_val = build_partition(val_files, "VALIDATION")
    X_test, y_test = build_partition(test_files, "UNSEEN TEST")

    print("\nShapes:")
    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Unseen test:", X_test.shape)

    print("\n===== STEP 3: SCALE DATA =====")
    X_train, X_val, X_test, scaler = scale_data(
        X_train, X_val, X_test
    )

    print("\n===== STEP 4: TRAIN NEW LSTM =====")
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    model = build_lstm_model()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=0.00001
        )
    ]

    history = model.fit(
        X_train,
        make_targets(y_train),
        validation_data=(X_val, make_targets(y_val)),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    print("\n===== STEP 5: EVALUATE UNSEEN ALGORITHMS =====")
    predictions = model.predict(X_test, verbose=0)
    results = evaluate(y_test, predictions)

    results["test_algorithms"] = sorted(held_out)
    results["test_samples"] = int(len(X_test))
    results["training_algorithms"] = sorted(
        available - held_out
    )

    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    model.save("models/unseen_algorithm_lstm.keras")
    joblib.dump(scaler, "models/unseen_algorithm_scaler.pkl")

    with open("results/unseen_algorithm_metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    with open("results/unseen_algorithm_history.json", "w") as f:
        json.dump(history.history, f, indent=2)

    print("\n===== UNSEEN ALGORITHM RESULTS =====")
    print(json.dumps(results, indent=4))

    elapsed = (time.time() - start) / 60
    print("\nSaved model: models/unseen_algorithm_lstm.keras")
    print("Saved scaler: models/unseen_algorithm_scaler.pkl")
    print("Saved metrics: results/unseen_algorithm_metrics.json")
    print(f"Execution time: {elapsed:.2f} minutes")
    print("UNSEEN ALGORITHM TEST COMPLETED.")


if __name__ == "__main__":
    main()
