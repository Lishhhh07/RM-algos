import time
import numpy as np
import tensorflow as tf

from .data_loader import load_dataset
from .train import train_model
from .evaluate import evaluate_model


def main():
    start = time.time()

    print("TensorFlow:", tf.__version__)
    print("Devices:", tf.config.list_physical_devices())

    print("\n===== STEP 1: LOAD AND SPLIT DATA =====")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_dataset()

    print("\nSequence shapes:")
    print("Train:", X_train.shape, y_train.shape)
    print("Validation:", X_val.shape, y_val.shape)
    print("Test:", X_test.shape, y_test.shape)

    print("\n===== STEP 2: TRAIN V2 MULTI-TASK LSTM =====")
    model, history, input_scaler, target_scaler = train_model(
        X_train, y_train, X_val, y_val
    )

    print("\n===== STEP 3: EVALUATE ON HELD-OUT FILES (RANDOM FILE-LEVEL SPLIT) =====")
    X_test_scaled = input_scaler.transform(
        X_test.reshape(-1, X_test.shape[-1])
    ).reshape(X_test.shape).astype(np.float32)

    evaluate_model(
        model,
        X_test_scaled,
        y_test,
        target_scaler
    )

    elapsed = (time.time() - start) / 60

    print(f"\nTOTAL EXECUTION TIME: {elapsed:.2f} minutes")
    print("V2 PIPELINE COMPLETED.")


if __name__ == "__main__":
    main()
