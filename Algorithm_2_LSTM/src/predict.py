import numpy as np
import tensorflow as tf
import joblib

from .config import (
    WINDOW_SIZE,
    FEATURES,
    MODEL_PATH_V2,
    SCALER_PATH_V2,
    TARGET_SCALER_PATH_V2
)


def load_trained_model():
    model = tf.keras.models.load_model(MODEL_PATH_V2)
    input_scaler = joblib.load(SCALER_PATH_V2)
    target_scaler = joblib.load(TARGET_SCALER_PATH_V2)

    return model, input_scaler, target_scaler


def predict_congestion(recent_measurements):
    model, input_scaler, target_scaler = load_trained_model()

    sequence = np.asarray(recent_measurements, dtype=np.float32)

    expected = (WINDOW_SIZE, len(FEATURES))

    if sequence.shape != expected:
        raise ValueError(f"Expected shape {expected}, got {sequence.shape}")

    scaled = input_scaler.transform(sequence).astype(np.float32)

    predictions = model.predict(
        scaled[np.newaxis, ...],
        verbose=0
    )

    rtt_scaled = np.asarray(
        predictions["rtt_delta"]
    ).reshape(-1, 1)

    rtt_delta = float(
        target_scaler.inverse_transform(rtt_scaled)[0, 0]
    )

    inflation = float(predictions["rtt_inflation"][0][0])
    retransmission = float(predictions["retransmission"][0][0])

    return {
        "predicted_rtt_delta": rtt_delta,
        "rtt_inflation_probability": inflation,
        "retransmission_probability": retransmission,
        "congestion_warning": inflation >= 0.5 or retransmission >= 0.5
    }
