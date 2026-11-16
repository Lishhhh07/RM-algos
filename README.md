# Predictive TCP Congestion Control Using Machine Learning

Course project - Algorithm 1: **Decision Tree Regressor for future-RTT prediction**.

Team: Lishika Meghani (1MS24CI065), Kanishk Tippimath (1MS24CI061), Janvi Jindal (1MS24CI058)

## What it does
Reads TCP `ss` measurements (RTT, cwnd, bytes sent/acked/retransmitted), builds sliding-window
features (last 10 samples), and trains a Decision Tree to predict the RTT **5 samples ahead**.
It is compared against a baseline that assumes "RTT stays the same".

## Dataset
Data Center TCP dataset (Zenodo, Data in Brief 2024, DOI: 10.1016/j.dib.2024.110522).
850 CSV files, 17 TCP variants, ~3.09 million rows. **Not included in this repo** (too large) -
download it and place it in a folder named `tcp-dataset/` with one sub-folder per variant.

## Run
```bash
pip install -r requirements.txt
python rtt_predictor.py --data tcp-dataset
```
Outputs go to `rtt_output/` (model, metrics.json, per-variant CSV, plots).
Options: `--horizon 5 --window 10 --stride 5` (use `--stride 1` for every window, slower).

## Results (unseen test flows, RTT in ms)
| | MAE | RMSE | R2 |
|---|---|---|---|
| Decision Tree | 8.6 | 15.9 | 0.90 |
| "RTT stays the same" baseline | 24.8 | 35.0 | 0.51 |

Congestion = RTT rises >20% above the flow's recent median within 5 samples:
tree precision 0.83 / recall 0.67 vs reactive baseline 0.35 / 0.24.
Model size 374 KB, 0.045 ms per prediction.
