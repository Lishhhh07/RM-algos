# Algorithm 2: Multi-Task LSTM for Predictive TCP Congestion Control

## 1. Problem Statement

TCP congestion can lead to increased Round-Trip Time (RTT), packet retransmissions and reduced network performance. Identifying these changes early can help in making better decisions about network congestion.

The aim of this project is to develop a multi-task LSTM model that learns from historical TCP network measurements and predicts three congestion-related parameters:

1. Future change in RTT.
2. Whether significant RTT inflation is likely to occur.
3. The risk of packet retransmissions.

The idea is to use a single model to perform all three prediction tasks instead of building three separate models.

## 2. Project Overview

The project uses an LSTM (Long Short-Term Memory) neural network to learn patterns from sequential TCP data.

The model takes the previous 10 samples as input and predicts congestion-related behaviour 5 samples ahead.

It has three output heads:

- **RTT Delta:** A regression output that predicts the change in RTT.
- **RTT Inflation:** A classification output that predicts whether RTT inflation will occur.
- **Retransmission Risk:** A classification output that predicts the possibility of retransmissions.

The model is trained and evaluated using data from 17 TCP congestion-control algorithms.

## 3. Directory Structure

```text
Algorithm_2/
│
├── data/
│   ├── bbr/
│   ├── bic/
│   ├── cubic/
│   ├── reno/
│   └── ... (17 TCP algorithms)
│
├── models/
│   ├── lstm_congestion_model.keras
│   └── scaler.pkl
│
├── results/
│   ├── evaluation_metrics.json
│   ├── predictions.csv
│   ├── training_history.json
│   └── README.md
│
├── plots/
│   ├── training_validation_loss.png
│   ├── rtt_actual_vs_predicted.png
│   ├── rtt_inflation_confusion_matrix.png
│   ├── retransmission_confusion_matrix.png
│   └── classification_performance.png
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── main.py
│   └── plot_results.py
│
├── requirements.txt
└── README.md
```

## 4. Description of Files

### Source Code (`src/`)

| File | Purpose |
|---|---|
| `config.py` | Stores model settings, feature names, training parameters and file paths. |
| `data_loader.py` | Reads the TCP CSV files, cleans the data and divides the files into training, validation and testing sets. |
| `feature_engineering.py` | Creates input sequences and generates the three prediction targets. |
| `model.py` | Defines the shared LSTM architecture and its three output layers. |
| `train.py` | Scales the input features, trains the model and saves the trained model and training history. |
| `evaluate.py` | Evaluates model performance using regression and classification metrics. |
| `predict.py` | Generates predictions using the trained model. |
| `main.py` | Runs the complete project pipeline. |
| `plot_results.py` | Generates graphs from the saved training history, predictions and evaluation results. |

### Trained Model (`models/`)

| File | Purpose |
|---|---|
| `lstm_congestion_model.keras` | Contains the trained LSTM model and its learned parameters. |
| `scaler.pkl` | Stores the feature scaler used during training so the input data can be transformed consistently during prediction. |

### Results (`results/`)

| File | Purpose |
|---|---|
| `evaluation_metrics.json` | Stores the final evaluation scores for all three prediction tasks. |
| `predictions.csv` | Contains the actual values and model predictions for the test dataset. |
| `training_history.json` | Stores training and validation metrics for each epoch. |
| `README.md` | Contains a brief description of the saved results. |

### Visualizations (`plots/`)

| File | Purpose |
|---|---|
| `training_validation_loss.png` | Shows how training and validation loss change across epochs. |
| `rtt_actual_vs_predicted.png` | Compares actual RTT changes with the values predicted by the model. |
| `rtt_inflation_confusion_matrix.png` | Shows correct and incorrect RTT inflation classifications. |
| `retransmission_confusion_matrix.png` | Shows correct and incorrect retransmission-risk classifications. |
| `classification_performance.png` | Compares accuracy, precision, recall, F1 score and PR-AUC for both classification tasks. |

## 5. How to Run the Algorithm

### Step 1: Set up the environment

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

### Step 2: Install the required libraries

```bash
pip install -r requirements.txt
```

### Step 3: Add the dataset

Place the dataset inside the `data/` directory. Keep the CSV files organised into folders according to their TCP congestion-control algorithm.

### Step 4: Run the complete pipeline

From the main project directory, execute:

```bash
python -m src.main
```

This runs the data loading, preprocessing, sequence generation, model training and evaluation stages.

The trained model and evaluation outputs are saved automatically.

### Step 5: Generate visualizations

After the pipeline finishes, execute:

```bash
python -m src.plot_results
```

The generated graphs are saved inside the `plots/` directory.

## 6. Results Obtained

The model was evaluated on 49,544 test samples.

### A. RTT Delta Prediction

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 7.9953 |
| Root Mean Squared Error (RMSE) | 15.8411 |
| R² Score | 0.7915 |

### B. RTT Inflation Detection

| Metric | Result |
|---|---:|
| Accuracy | 94.13% |
| Precision | 86.20% |
| Recall | 74.12% |
| F1 Score | 79.70% |
| PR-AUC | 88.64% |

### C. Retransmission Risk Prediction

| Metric | Result |
|---|---:|
| Accuracy | 94.16% |
| Precision | 89.42% |
| Recall | 88.00% |
| F1 Score | 88.70% |
| PR-AUC | 96.20% |

The results show that the model can identify retransmission-related patterns with good initial performance. RTT inflation detection also achieved high accuracy, although its recall indicates that some inflation events were missed.

## 7. Future Scope

The current model predicts congestion-related behaviour using historical TCP measurements. These predictions can be used as the basis for a future congestion-control mechanism.

Possible improvements include:

- Using predicted congestion signals to adjust the TCP congestion window.
- Comparing the model with traditional TCP algorithms such as Reno and CUBIC.
- Testing the approach under different network conditions using Mininet.
- Evaluating throughput, latency, packet loss and fairness.
- Testing the model on previously unseen TCP algorithms.

