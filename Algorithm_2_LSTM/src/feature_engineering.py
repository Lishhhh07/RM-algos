import numpy as np

from .config import (
    WINDOW_SIZE,
    PREDICTION_HORIZON,
    STRIDE,
    RTT_INFLATION_FACTOR,
    FEATURES
)


def create_sequences(df):
    required = FEATURES + ["rtt", "retransmission_ratio"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    values = df[FEATURES].to_numpy(dtype=np.float32)
    rtt = df["rtt"].to_numpy(dtype=np.float32)
    retrans = df["retransmission_ratio"].to_numpy(dtype=np.float32)

    X, y = [], []
    end = len(df) - PREDICTION_HORIZON + 1

    for i in range(WINDOW_SIZE, end, STRIDE):
        current_rtt = rtt[i - 1]
        future_rtt = rtt[i + PREDICTION_HORIZON - 1]

        rtt_delta = future_rtt - current_rtt
        rtt_inflation = int(
            future_rtt > RTT_INFLATION_FACTOR * current_rtt
        )

        future_retrans = retrans[i:i + PREDICTION_HORIZON]
        retransmission = int(np.any(future_retrans > 0))

        X.append(values[i - WINDOW_SIZE:i])
        y.append([rtt_delta, rtt_inflation, retransmission])

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32)
    )
