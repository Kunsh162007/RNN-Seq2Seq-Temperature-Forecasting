"""
train.py
────────
Trains all 7 RNN models sequentially and saves:
  • Keras model checkpoints  →  models/<name>.keras
  • Combined results JSON    →  results/all_results.json

Usage
─────
    python train.py                   # train all models
    python train.py --models LSTM GRU # train specific models only
    python train.py --epochs 50       # override epoch count
"""

import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf

# ── make src importable regardless of working directory ───────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from src.config import (
    DATA_FILE, SEQ_LEN, PRED_LEN, TEST_RATIO, VAL_RATIO,
    BATCH_SIZE, EPOCHS, PATIENCE, SEED, UNITS, DROPOUT,
    MODELS_DIR, RESULTS_DIR, MODEL_NAMES
)
from src.utils  import load_data, prepare_datasets, compute_metrics, save_results
from src.models import get_model, get_attention_weights

# ── reproducibility ───────────────────────────────────────────────────────────
np.random.seed(SEED)
tf.random.set_seed(SEED)


def get_callbacks(name: str) -> list:
    os.makedirs(MODELS_DIR, exist_ok=True)
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=6,
            min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODELS_DIR, f"{name.replace(' ', '_')}.keras"),
            save_best_only=True, monitor="val_loss", verbose=0
        ),
    ]


def train_all(model_names: list, epochs: int) -> None:
    # ── Load & preprocess data ─────────────────────────────────────────────
    print("\n" + "="*60)
    print(" Loading & preprocessing data...")
    print("="*60)
    df   = load_data(DATA_FILE)
    data = prepare_datasets(df, SEQ_LEN, PRED_LEN, VAL_RATIO, TEST_RATIO)

    scaler               = data["scaler"]
    X_tr, y_tr, X_v, y_v, X_te, y_te     = data["single"]
    X_s_tr, y_s_tr, X_s_v, y_s_v, X_s_te, y_s_te = data["multi"]
    n_tr, n_val, n_test  = data["split_idx"]

    print(f"  Train   : {n_tr:,}  |  Val : {n_val:,}  |  Test : {n_test:,}")
    print(f"  X shape : {X_tr.shape}  →  y shape : {y_tr.shape}")

    all_results = {}

    for name in model_names:
        is_seq2seq = (name == "Seq2Seq")
        print(f"\n{'━'*60}")
        print(f"  [{model_names.index(name)+1}/{len(model_names)}]  {name}")
        print(f"{'━'*60}")

        model = get_model(name, SEQ_LEN, PRED_LEN, UNITS, DROPOUT)
        model.summary()

        # ── Select appropriate data ────────────────────────────────────────
        if is_seq2seq:
            Xtr, ytr = X_s_tr, y_s_tr[..., np.newaxis]
            Xv,  yv  = X_s_v,  y_s_v[..., np.newaxis]
            Xte, yte = X_s_te, y_s_te
        else:
            Xtr, ytr = X_tr, y_tr
            Xv,  yv  = X_v,  y_v
            Xte, yte = X_te, y_te

        # ── Train ──────────────────────────────────────────────────────────
        history = model.fit(
            Xtr, ytr,
            validation_data=(Xv, yv),
            epochs=epochs,
            batch_size=BATCH_SIZE,
            callbacks=get_callbacks(name),
            verbose=1,
        )

        # ── Predict & invert scaling ───────────────────────────────────────
        y_pred_sc = model.predict(Xte, verbose=0)

        if is_seq2seq:
            # y_pred_sc: (N, pred_len, 1)  →  flatten for global metrics
            y_true = scaler.inverse_transform(yte.reshape(-1,1)).flatten()
            y_pred = scaler.inverse_transform(
                y_pred_sc.reshape(-1,1)).flatten()

            # Also save per-step preds for the multi-step visualisation
            # shape: (N, pred_len)
            y_pred_steps = scaler.inverse_transform(
                y_pred_sc.squeeze(-1).reshape(-1,1)
            ).reshape(-1, PRED_LEN)
            y_true_steps = scaler.inverse_transform(
                yte.reshape(-1,1)
            ).reshape(-1, PRED_LEN)
        else:
            y_true = scaler.inverse_transform(yte.reshape(-1,1)).flatten()
            y_pred = scaler.inverse_transform(y_pred_sc.reshape(-1,1)).flatten()

        metrics = compute_metrics(y_true, y_pred)
        n_params = model.count_params()
        ep_done  = len(history.history["loss"])

        all_results[name] = {
            "metrics"  : metrics,
            "history"  : history.history,
            "y_true"   : y_true,
            "y_pred"   : y_pred,
            "n_params" : n_params,
            "epochs"   : ep_done,
        }

        # Seq2Seq: save per-step arrays for multi-step forecast plot
        if is_seq2seq:
            all_results[name]["y_pred_steps"] = y_pred_steps
            all_results[name]["y_true_steps"] = y_true_steps

        # Attention weights (for Attention LSTM only)
        if name == "Attention LSTM":
            weights = get_attention_weights(model, Xte[:100])
            all_results[name]["attn_weights"] = weights   # (100, seq_len)

        print(f"\n  ✅  {name}  →  MAE:{metrics['MAE']:.4f}  "
              f"RMSE:{metrics['RMSE']:.4f}  "
              f"MAPE:{metrics['MAPE']:.2f}%  "
              f"R²:{metrics['R2']:.4f}  "
              f"({n_params:,} params,  {ep_done} epochs)")

    # ── Save all results ───────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_path = os.path.join(RESULTS_DIR, "all_results.json")

    # Custom serialiser handles numpy arrays in attn_weights / step preds
    serialisable = {}
    for mn, v in all_results.items():
        entry = {
            "metrics"  : v["metrics"],
            "history"  : {k: [float(x) for x in vals]
                          for k, vals in v["history"].items()},
            "y_true"   : v["y_true"].tolist(),
            "y_pred"   : v["y_pred"].tolist(),
            "n_params" : int(v["n_params"]),
            "epochs"   : int(v["epochs"]),
        }
        if "attn_weights"  in v: entry["attn_weights"]  = v["attn_weights"].tolist()
        if "y_pred_steps"  in v: entry["y_pred_steps"]  = v["y_pred_steps"].tolist()
        if "y_true_steps"  in v: entry["y_true_steps"]  = v["y_true_steps"].tolist()
        serialisable[mn] = entry

    with open(result_path, "w") as f:
        json.dump(serialisable, f, indent=2)

    # ── Print final leaderboard ────────────────────────────────────────────
    print("\n" + "="*60)
    print("  FINAL LEADERBOARD  (sorted by RMSE)")
    print("="*60)
    print(f"  {'Model':<24} {'MAE':>6} {'RMSE':>6} {'MAPE%':>7} {'R²':>6}  Params")
    print("  " + "-"*56)
    sorted_models = sorted(all_results.items(),
                           key=lambda x: x[1]["metrics"]["RMSE"])
    for mn, v in sorted_models:
        m = v["metrics"]
        print(f"  {mn:<24} {m['MAE']:>6.4f} {m['RMSE']:>6.4f} "
              f"{m['MAPE']:>6.2f}%  {m['R2']:>6.4f}  {v['n_params']:,}")
    print("="*60)
    print(f"\n  Results saved → {result_path}")
    print("  Run  python visualize.py  to generate all plots.\n")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RNN models — Week 7")
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES,
                        help="Which models to train (default: all)")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Max epochs (default: {EPOCHS})")
    args = parser.parse_args()

    # Validate requested model names
    invalid = [m for m in args.models if m not in MODEL_NAMES]
    if invalid:
        print(f"❌  Unknown model(s): {invalid}")
        print(f"    Available: {MODEL_NAMES}")
        sys.exit(1)

    train_all(args.models, args.epochs)
