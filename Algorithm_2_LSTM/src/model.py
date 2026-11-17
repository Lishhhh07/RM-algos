import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from .config import (
    WINDOW_SIZE,
    FEATURES,
    LSTM_UNITS,
    DROPOUT_RATE,
    LEARNING_RATE
)


def build_lstm_model():
    inputs = Input(
        shape=(WINDOW_SIZE, len(FEATURES)),
        name="network_sequence"
    )

    x = LSTM(LSTM_UNITS, name="shared_lstm")(inputs)
    x = Dropout(DROPOUT_RATE)(x)

    rtt_delta = Dense(1, activation="linear", name="rtt_delta")(x)
    rtt_inflation = Dense(1, activation="sigmoid", name="rtt_inflation")(x)
    retransmission = Dense(1, activation="sigmoid", name="retransmission")(x)

    model = Model(
        inputs=inputs,
        outputs={
            "rtt_delta": rtt_delta,
            "rtt_inflation": rtt_inflation,
            "retransmission": retransmission
        },
        name="multi_task_tcp_lstm"
    )

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss={
            "rtt_delta": tf.keras.losses.Huber(),
            "rtt_inflation": "binary_crossentropy",
            "retransmission": "binary_crossentropy"
        },
        metrics={
            "rtt_delta": ["mae"],
            "rtt_inflation": [
                "accuracy", tf.keras.metrics.AUC(name="auc")
            ],
            "retransmission": [
                "accuracy", tf.keras.metrics.AUC(name="auc")
            ]
        }
    )

    return model
