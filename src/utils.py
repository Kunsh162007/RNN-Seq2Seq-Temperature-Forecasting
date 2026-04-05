"""
src/utils.py
────────────
Data loading, sliding-window creation, scaling, and metric helpers.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Load CSV → DataFrame with a DatetimeIndex and a single 'Temp' column."""
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    df.columns = ["Temp"]
    return df


# ── Sliding-window conversion ──────────────────────────────────────────────────

def create_sequences(data: np.ndarray, seq_len: int, pred_len: int = 1):
    """
    Convert a 1-D scaled array into supervised (X, y) pairs.

    Returns
    -------
    X : (N, seq_len, 1)
    y : (N,)           if pred_len == 1
        (N, pred_len)  if pred_len > 1
    """
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i : i + seq_len])
        if pred_len == 1:
            y.append(data[i + seq_len])
        else:
            y.append(data[i + seq_len : i + seq_len + pred_len])
    X = np.array(X)[..., np.newaxis]   # (N, T, 1)
    y = np.array(y)
    return X, y


# ── Full preprocessing pipeline ────────────────────────────────────────────────

def prepare_datasets(df: pd.DataFrame, seq_len: int, pred_len: int,
                     val_ratio: float, test_ratio: float):
    """
    Scale → create sequences → split into train / val / test.

    Returns a dict with keys:
        scaler,
        single: (X_train, y_train, X_val, y_val, X_test, y_test),
        multi:  (X_train, y_train, X_val, y_val, X_test, y_test),
        split_idx: (n_train, n_val, n_test)
    """
    scaler = MinMaxScaler((0, 1))
    scaled = scaler.fit_transform(df.values).flatten()

    # Single-step windows (for all models except Seq2Seq)
    Xs, ys = create_sequences(scaled, seq_len, pred_len=1)

    # Multi-step windows (for Seq2Seq)
    Xm, ym = create_sequences(scaled, seq_len, pred_len=pred_len)

    n      = len(Xs)
    n_test = int(n * test_ratio)
    n_val  = int(n * val_ratio)
    n_tr   = n - n_test - n_val

    def split(X, y):
        return (X[:n_tr], y[:n_tr],
                X[n_tr:n_tr+n_val], y[n_tr:n_tr+n_val],
                X[n_tr+n_val:], y[n_tr+n_val:])

    return {
        "scaler"    : scaler,
        "single"    : split(Xs, ys),
        "multi"     : split(Xm, ym),
        "split_idx" : (n_tr, n_val, n_test),
    }


# ── Metric helpers ─────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return MAE, RMSE, MAPE, R² as a dict."""
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {
        "MAE"  : float(round(mean_absolute_error(y_true, y_pred), 4)),
        "RMSE" : float(round(np.sqrt(mean_squared_error(y_true, y_pred)), 4)),
        "MAPE" : float(round(mape, 4)),
        "R2"   : float(round(r2_score(y_true, y_pred), 4)),
    }


# ── Persistence helpers ────────────────────────────────────────────────────────

def save_results(results: dict, path: str) -> None:
    """Save metrics + history + predictions to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    serialisable = {}
    for model_name, v in results.items():
        serialisable[model_name] = {
            "metrics"  : v["metrics"],
            "history"  : {k: [float(x) for x in vals]
                          for k, vals in v["history"].items()},
            "y_true"   : v["y_true"].tolist(),
            "y_pred"   : v["y_pred"].tolist(),
            "n_params" : int(v["n_params"]),
            "epochs"   : int(v["epochs"]),
        }

    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"  Results saved → {path}")


def load_results(path: str) -> dict:
    """Load results JSON saved by save_results()."""
    with open(path) as f:
        raw = json.load(f)
    results = {}
    for name, v in raw.items():
        results[name] = {
            "metrics"  : v["metrics"],
            "history"  : v["history"],
            "y_true"   : np.array(v["y_true"]),
            "y_pred"   : np.array(v["y_pred"]),
            "n_params" : v["n_params"],
            "epochs"   : v["epochs"],
        }
    return results
