RANDOM_SEED = 42

WINDOW_SIZE = 10
PREDICTION_HORIZON = 5
STRIDE = 10

FEATURES = [
    "rtt",
    "cwnd",
    "bytes_acked_delta",
    "retransmission_ratio"
]

RTT_INFLATION_FACTOR = 1.2
MIN_FLOW_ROWS = 60

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

LSTM_UNITS = 64
DROPOUT_RATE = 0.2

EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 4

MODEL_PATH = "models/lstm_congestion_model.keras"
SCALER_PATH = "models/scaler.pkl"
METRICS_PATH = "results/evaluation_metrics.json"
PREDICTIONS_PATH = "results/predictions.csv"
HISTORY_PATH = "results/training_history.json"

# V2 model and output paths
MODEL_PATH_V2 = "models/lstm_congestion_model_v2.keras"
SCALER_PATH_V2 = "models/scaler_v2.pkl"
TARGET_SCALER_PATH_V2 = "models/rtt_target_scaler_v2.pkl"

METRICS_PATH_V2 = "results/evaluation_metrics_v2.json"
PREDICTIONS_PATH_V2 = "results/predictions_v2.csv"
HISTORY_PATH_V2 = "results/training_history_v2.json"
