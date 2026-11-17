import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parent.parent
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)

with open(ROOT / "results/training_history_v2.json") as f:
    history = json.load(f)

with open(ROOT / "results/evaluation_metrics_v2.json") as f:
    metrics = json.load(f)

df = pd.read_csv(ROOT / "results/predictions_v2.csv")

# 1. Training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history["loss"], label="Training Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS / "training_validation_loss.png", dpi=200)
plt.close()

# 2. Actual vs predicted RTT delta
actual = df["actual_rtt_delta"].to_numpy()
predicted = df["predicted_rtt_delta"].to_numpy()

rng = np.random.default_rng(42)
indices = rng.choice(len(actual), min(3000, len(actual)), replace=False)

plt.figure(figsize=(7, 7))
plt.scatter(actual[indices], predicted[indices], alpha=0.3, s=10)
low = min(actual.min(), predicted.min())
high = max(actual.max(), predicted.max())
plt.plot([low, high], [low, high], "r--", label="Perfect Prediction")
plt.xlabel("Actual RTT Delta")
plt.ylabel("Predicted RTT Delta")
plt.title("Actual vs Predicted RTT Delta (3,000 Sample)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS / "rtt_actual_vs_predicted.png", dpi=200)
plt.close()

# 3. RTT inflation confusion matrix
y_true = df["actual_rtt_inflation"].astype(int)
y_pred = (
    df["predicted_rtt_inflation_probability"] >= 0.5
).astype(int)

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Inflation", "Inflation"]
)
disp.plot(cmap="Blues", values_format="d")
plt.title("RTT Inflation Confusion Matrix")
plt.tight_layout()
plt.savefig(PLOTS / "rtt_inflation_confusion_matrix.png", dpi=200)
plt.close()

# 4. Retransmission confusion matrix
y_true = df["actual_retransmission"].astype(int)
y_pred = (
    df["predicted_retransmission_probability"] >= 0.5
).astype(int)

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Retransmission", "Retransmission"]
)
disp.plot(cmap="Blues", values_format="d")
plt.title("Retransmission Risk Confusion Matrix")
plt.tight_layout()
plt.savefig(PLOTS / "retransmission_confusion_matrix.png", dpi=200)
plt.close()

# 5. Consolidated classification performance
labels = ["Accuracy", "Precision", "Recall", "F1", "PR-AUC"]

tasks = {
    "RTT Inflation": metrics["RTT Inflation"],
    "Retransmission Risk": metrics["Retransmission Risk"]
}

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(10, 5))

for i, (name, values) in enumerate(tasks.items()):
    scores = [
        values["accuracy"],
        values["precision"],
        values["recall"],
        values["f1"],
        values["pr_auc"]
    ]
    plt.bar(x + (i - 0.5) * width, scores, width, label=name)

plt.xticks(x, labels)
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Classification Performance Comparison")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS / "classification_performance.png", dpi=200)
plt.close()

print("\nALL PLOTS GENERATED SUCCESSFULLY")
for path in sorted(PLOTS.glob("*.png")):
    print(path.name, "-", f"{path.stat().st_size / 1024:.1f} KB")
