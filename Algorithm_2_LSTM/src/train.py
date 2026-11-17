import json
import numpy as np
import tensorflow as tf
import joblib

from pathlib import Path
from sklearn.preprocessing import StandardScaler

from .model import build_lstm_model
from .config import (
    EPOCHS,
    BATCH_SIZE,
    RANDOM_SEED,
    EARLY_STOPPING_PATIENCE,
    MODEL_PATH_V2,
    SCALER_PATH_V2,
    TARGET_SCALER_PATH_V2,
    HISTORY_PATH_V2
)


def make_targets(y, target_scaler):
    return {
        "rtt_delta": target_scaler.transform(
            y[:, 0:1]
        ).astype(np.float32),
        "rtt_inflation": y[:, 1:2],
        "retransmission": y[:, 2:3]
    }


def train_model(X_train, y_train, X_val, y_val):
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    input_scaler = StandardScaler()
    target_scaler = StandardScaler()

    X_train_scaled = input_scaler.fit_transform(
        X_train.reshape(-1, X_train.shape[-1])
    ).reshape(X_train.shape).astype(np.float32)

    X_val_scaled = input_scaler.transform(
        X_val.reshape(-1, X_val.shape[-1])
    ).reshape(X_val.shape).astype(np.float32)

    target_scaler.fit(y_train[:, 0:1])

    y_train_targets = make_targets(y_train, target_scaler)
    y_val_targets = make_targets(y_val, target_scaler)

    print("\nRTT target scaling:")
    print("Training mean:", float(target_scaler.mean_[0]))
    print("Training standard deviation:", float(target_scaler.scale_[0]))

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
        X_train_scaled,
        y_train_targets,
        validation_data=(X_val_scaled, y_val_targets),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    model.save(MODEL_PATH_V2)
    joblib.dump(input_scaler, SCALER_PATH_V2)
    joblib.dump(target_scaler, TARGET_SCALER_PATH_V2)

    with open(HISTORY_PATH_V2, "w") as f:
        json.dump(history.history, f, indent=2)

    print("\nV2 model saved:", MODEL_PATH_V2)
    print("Input scaler saved:", SCALER_PATH_V2)
    print("RTT target scaler saved:", TARGET_SCALER_PATH_V2)
    print("Training history saved:", HISTORY_PATH_V2)

    return model, history, input_scaler, target_scaler
