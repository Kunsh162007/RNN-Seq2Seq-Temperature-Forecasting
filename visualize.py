"""
visualize.py
────────────
Generates 8 premium publication-quality plots from saved training results.

Usage
─────
    python visualize.py                    # generate all plots
    python visualize.py --plot dashboard   # single plot
    python visualize.py --dpi 300          # set DPI (default: 200)

Available plots
───────────────
    dashboard     — hero multi-panel overview (LinkedIn banner)
    metrics       — grouped bar chart: MAE / RMSE / MAPE / R²
    loss_curves   — training & validation loss for every model
    predictions   — actual vs predicted overlay (all 7 models)
    radar         — spider chart comparing normalised metrics
    scatter       — complexity (params) vs RMSE
    attention     — attention-weight heatmap over time steps
    seq2seq       — 7-day multi-step forecast showcase
    gradient      — BPTT vanishing/exploding gradient illustration
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(__file__))
from src.config import (
    PALETTE, BG_DARK, BG_PANEL, BG_GRID, FG_TEXT, FG_MUTED, ACCENT,
    RESULTS_DIR, PLOTS_DIR, SEQ_LEN, PRED_LEN, MODEL_NAMES
)

RESULT_FILE = os.path.join(RESULTS_DIR, "all_results.json")


# ─────────────────────────────────────────────────────────────────────────────
#  Global matplotlib style
# ─────────────────────────────────────────────────────────────────────────────
def _apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor"     : BG_DARK,
        "axes.facecolor"       : BG_PANEL,
        "axes.edgecolor"       : BG_GRID,
        "axes.labelcolor"      : FG_TEXT,
        "axes.titlecolor"      : FG_TEXT,
        "axes.grid"            : True,
        "axes.spines.top"      : False,
        "axes.spines.right"    : False,
        "grid.color"           : BG_GRID,
        "grid.linewidth"       : 0.6,
        "text.color"           : FG_TEXT,
        "xtick.color"          : FG_MUTED,
        "ytick.color"          : FG_MUTED,
        "xtick.labelsize"      : 8,
        "ytick.labelsize"      : 8,
        "legend.facecolor"     : BG_PANEL,
        "legend.edgecolor"     : BG_GRID,
        "legend.labelcolor"    : FG_TEXT,
        "legend.fontsize"      : 8,
        "font.family"          : "DejaVu Sans",
        "figure.dpi"           : 150,
        "savefig.facecolor"    : BG_DARK,
        "savefig.bbox"         : "tight",
        "savefig.pad_inches"   : 0.15,
    })


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _load() -> dict:
    if not os.path.exists(RESULT_FILE):
        print(f"❌  Results file not found: {RESULT_FILE}")
        print("    Run  python train.py  first.")
        sys.exit(1)
    with open(RESULT_FILE) as f:
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
        if "attn_weights"  in v: results[name]["attn_weights"]  = np.array(v["attn_weights"])
        if "y_pred_steps"  in v: results[name]["y_pred_steps"]  = np.array(v["y_pred_steps"])
        if "y_true_steps"  in v: results[name]["y_true_steps"]  = np.array(v["y_true_steps"])
    return results


def _save(fig: plt.Figure, name: str, dpi: int) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, f"{name}.png")
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
    print(f"  ✅  Saved → {path}")
    plt.close(fig)


def _glow(ax, x, y, color, lw=2.0, alpha_main=1.0, n_glow=4):
    """Draw a line with a glowing halo effect."""
    for i in range(n_glow, 0, -1):
        ax.plot(x, y, color=color, lw=lw + i * 1.8,
                alpha=0.04, solid_capstyle="round")
    ax.plot(x, y, color=color, lw=lw, alpha=alpha_main,
            solid_capstyle="round")


def _badge(ax, x, y, text, color):
    """Small coloured badge annotation."""
    ax.annotate(text, (x, y),
                fontsize=7, fontweight="bold", color=BG_DARK,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc=color,
                          ec="none", alpha=0.92))


def _section_title(fig, x, y, text, size=11):
    fig.text(x, y, text, fontsize=size, fontweight="bold",
             color=FG_TEXT, transform=fig.transFigure)


# ─────────────────────────────────────────────────────────────────────────────
#  1  DASHBOARD  — hero plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_dashboard(results: dict, dpi: int) -> None:
    """
    2-row × 4-col hero figure:
      Row 1: RMSE bar | Loss curves (2 selected) | Complexity scatter
      Row 2: Predictions (best model) | Radar chart
    """
    models = list(results.keys())
    colors = [PALETTE[m] for m in models]

    fig = plt.figure(figsize=(20, 10), facecolor=BG_DARK)
    fig.text(0.5, 0.965,
             "RNN Architecture Comparison  —  Daily Temperature Forecasting",
             ha="center", fontsize=16, fontweight="bold", color=FG_TEXT)
    fig.text(0.5, 0.945,
             "Week 7 · Deep Learning · Sequence-to-Sequence Models",
             ha="center", fontsize=10, color=ACCENT)

    gs = gridspec.GridSpec(2, 4, figure=fig,
                           hspace=0.42, wspace=0.38,
                           left=0.06, right=0.97, top=0.90, bottom=0.07)

    # ── Panel A: RMSE bar chart ──────────────────────────────────────────
    ax_rmse = fig.add_subplot(gs[0, 0])
    rmse_vals = [results[m]["metrics"]["RMSE"] for m in models]
    bars = ax_rmse.barh(models, rmse_vals, color=colors,
                        edgecolor=BG_DARK, linewidth=0.8, height=0.6)
    for bar, val in zip(bars, rmse_vals):
        ax_rmse.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                     f"{val:.3f}", va="center", fontsize=7.5, color=FG_TEXT)
    ax_rmse.set_title("RMSE  (lower = better)", fontsize=9, pad=8)
    ax_rmse.invert_yaxis()
    ax_rmse.set_facecolor(BG_PANEL)

    # ── Panel B: Loss curves (best + worst) ──────────────────────────────
    ax_loss = fig.add_subplot(gs[0, 1:3])
    for name in models:
        hist = results[name]["history"]
        c    = PALETTE[name]
        ep   = range(1, len(hist["loss"]) + 1)
        _glow(ax_loss, ep, hist["loss"],     c, lw=1.6, alpha_main=0.9)
        ax_loss.plot(ep, hist["val_loss"], color=c, lw=1.0,
                     linestyle="--", alpha=0.55)
    ax_loss.set_title("Training Loss Curves  (— train  ╌╌ val)",
                      fontsize=9, pad=8)
    ax_loss.set_xlabel("Epoch", fontsize=8)
    ax_loss.set_ylabel("MSE Loss", fontsize=8)

    # Legend
    for name in models:
        ax_loss.plot([], [], color=PALETTE[name], lw=2, label=name)
    ax_loss.legend(loc="upper right", ncol=2, fontsize=7)

    # ── Panel C: Complexity scatter ───────────────────────────────────────
    ax_sc = fig.add_subplot(gs[0, 3])
    for name in models:
        p  = results[name]["n_params"] / 1000
        r  = results[name]["metrics"]["RMSE"]
        c  = PALETTE[name]
        ax_sc.scatter(p, r, color=c, s=110, zorder=5,
                      edgecolors=BG_DARK, linewidths=0.8)
        ax_sc.annotate(name.replace(" ", "\n"),
                       (p, r), fontsize=6, color=c,
                       xytext=(5, 4), textcoords="offset points")
    ax_sc.set_title("Params (k)  vs  RMSE", fontsize=9, pad=8)
    ax_sc.set_xlabel("Parameters (thousands)", fontsize=8)
    ax_sc.set_ylabel("RMSE", fontsize=8)

    # ── Panel D: Predictions — best model ────────────────────────────────
    best = min(models, key=lambda m: results[m]["metrics"]["RMSE"])
    ax_pred = fig.add_subplot(gs[1, 0:2])
    N = 120
    yt = results[best]["y_true"][:N]
    yp = results[best]["y_pred"][:N]
    ax_pred.fill_between(range(N), yt, alpha=0.12, color=PALETTE[best])
    _glow(ax_pred, range(N), yt, FG_MUTED, lw=1.4, alpha_main=0.7, n_glow=0)
    _glow(ax_pred, range(N), yp, PALETTE[best], lw=2.0)
    ax_pred.set_title(f"Best Model: {best}  —  Actual vs Predicted",
                      fontsize=9, pad=8)
    ax_pred.set_xlabel("Test Day", fontsize=8)
    ax_pred.set_ylabel("Temperature (°C)", fontsize=8)
    ax_pred.plot([], [], color=FG_MUTED, lw=1.5, label="Actual")
    ax_pred.plot([], [], color=PALETTE[best], lw=2, label="Predicted")
    ax_pred.legend(fontsize=8)

    # ── Panel E: Radar / spider chart ────────────────────────────────────
    ax_radar = fig.add_subplot(gs[1, 2:4], polar=True)
    _draw_radar(ax_radar, results, models)

    _save(fig, "01_dashboard", dpi)


# ─────────────────────────────────────────────────────────────────────────────
#  2  METRICS BAR CHART
# ─────────────────────────────────────────────────────────────────────────────
def plot_metrics(results: dict, dpi: int) -> None:
    models    = list(results.keys())
    metrics   = ["MAE", "RMSE", "MAPE", "R2"]
    labels    = ["MAE", "RMSE", "MAPE (%)", "R²"]
    better    = ["↓ lower", "↓ lower", "↓ lower", "↑ higher"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 6), facecolor=BG_DARK)
    fig.suptitle("Model Performance Metrics — All 7 Architectures",
                 fontsize=14, fontweight="bold", color=FG_TEXT, y=1.01)

    for ax, metric, label, b in zip(axes, metrics, labels, better):
        vals   = [results[m]["metrics"][metric] for m in models]
        colors = [PALETTE[m] for m in models]
        sorted_pairs = sorted(zip(vals, models, colors), reverse=(metric == "R2"))
        sv, sm, sc = zip(*sorted_pairs)

        bars = ax.barh(sm, sv, color=sc, edgecolor=BG_DARK,
                       linewidth=0.8, height=0.65)
        for bar, val in zip(bars, sv):
            ax.text(val + max(sv) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8, color=FG_TEXT)
        ax.set_title(f"{label}\n{b}", fontsize=9, pad=6, color=FG_TEXT)
        ax.set_facecolor(BG_PANEL)
        ax.invert_yaxis()
        ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    _save(fig, "02_metrics", dpi)


# ─────────────────────────────────────────────────────────────────────────────
#  3  LOSS CURVES
# ─────────────────────────────────────────────────────────────────────────────
def plot_loss_curves(results: dict, dpi: int) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), facecolor=BG_DARK)
    fig.suptitle("Training & Validation Loss Curves",
                 fontsize=14, fontweight="bold", color=FG_TEXT)
    axes = axes.flatten()

    for i, name in enumerate(results):
        ax   = axes[i]
        hist = results[name]["history"]
        c    = PALETTE[name]
        ep   = range(1, len(hist["loss"]) + 1)

        # shaded area between train and val
        ax.fill_between(ep, hist["loss"], hist["val_loss"],
                        alpha=0.07, color=c)
        _glow(ax, ep, hist["loss"],     c, lw=2.2, alpha_main=1.0)
        ax.plot(ep, hist["val_loss"], color=c, lw=1.5,
                linestyle="--", alpha=0.65, label="Val")

        ax.set_title(name, fontsize=9, fontweight="bold", color=c)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("MSE Loss", fontsize=8)

        m = results[name]["metrics"]
        ax.text(0.97, 0.92,
                f"RMSE {m['RMSE']:.3f}\nR² {m['R2']:.3f}",
                transform=ax.transAxes, fontsize=7.5,
                color=c, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3",
                          fc=BG_DARK, ec=c, alpha=0.7))

        epochs_run = results[name]["epochs"]
        ax.text(0.03, 0.06, f"{epochs_run} epochs",
                transform=ax.transAxes, fontsize=7,
                color=FG_MUTED)
        ax.set_facecolor(BG_PANEL)

    axes[-1].set_visible(False)
    plt.tight_layout()
    _save(fig, "03_loss_curves", dpi)


# ─────────────────────────────────────────────────────────────────────────────
#  4  PREDICTIONS OVERLAY
# ─────────────────────────────────────────────────────────────────────────────
def plot_predictions(results: dict, dpi: int) -> None:
    N   = 150
    fig, axes = plt.subplots(4, 2, figsize=(18, 16), facecolor=BG_DARK)
    fig.suptitle("Actual vs Predicted Temperature  —  First 150 Test Days",
                 fontsize=14, fontweight="bold", color=FG_TEXT)
    axes = axes.flatten()

    for i, name in enumerate(results):
        ax  = axes[i]
        yt  = results[name]["y_true"][:N]
        yp  = results[name]["y_pred"][:N]
        c   = PALETTE[name]
        m   = results[name]["metrics"]

        ax.fill_between(range(N), yt, alpha=0.10, color=c)
        _glow(ax, range(N), yt, FG_MUTED, lw=1.5, alpha_main=0.55, n_glow=0)
        _glow(ax, range(N), yp, c, lw=2.0)

        ax.set_title(f"{name}", fontweight="bold", fontsize=10, color=c)
        ax.set_ylabel("Temp (°C)", fontsize=8)
        ax.set_facecolor(BG_PANEL)

        # Metric badge
        ax.text(0.97, 0.05,
                f"RMSE {m['RMSE']:.3f}  R² {m['R2']:.3f}",
                transform=ax.transAxes, fontsize=8,
                color=c, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3",
                          fc=BG_DARK, ec=c, alpha=0.75))

        ax.plot([], [], color=FG_MUTED, lw=1.5, label="Actual")
        ax.plot([], [], color=c, lw=2, label="Predicted")
        ax.legend(fontsize=8)

    axes[-1].set_visible(False)
    plt.tight_layout()
    _save(fig, "04_predictions", dpi)


# ─────────────────────────────────────────────────────────────────────────────
#  5  RADAR / SPIDER CHART  (standalone)
# ─────────────────────────────────────────────────────────────────────────────
def _draw_radar(ax, results: dict, models: list) -> None:
    """Draw radar chart on an existing polar axis."""
    # Normalise: lower MAE/RMSE/MAPE → higher score; higher R² → higher score
    raw = {m: [
        results[m]["metrics"]["MAE"],
        results[m]["metrics"]["RMSE"],
        results[m]["metrics"]["MAPE"],
        1 - results[m]["metrics"]["R2"],  # invert so higher = better
    ] for m in models}

    dims  = ["MAE", "RMSE", "MAPE", "1-R²"]
    n_dim = len(dims)
    angles = np.linspace(0, 2 * np.pi, n_dim, endpoint=False).tolist()
    angles += angles[:1]

    # Normalise 0-1 (0 = worst, 1 = best = lowest error)
    all_vals = np.array(list(raw.values()))
    mn = all_vals.min(axis=0)
    mx = all_vals.max(axis=0)
    eps = 1e-8

    for name in models:
        v     = np.array(raw[name])
        norm  = 1 - (v - mn) / (mx - mn + eps)  # invert: lower error → 1
        norm  = np.clip(norm, 0, 1).tolist() + [norm[0]]
        c     = PALETTE[name]
        ax.plot(angles, norm, color=c, lw=2.0, alpha=0.9)
        ax.fill(angles, norm, color=c, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, color=FG_TEXT, fontsize=8)
    ax.set_yticklabels([])
    ax.set_facecolor(BG_PANEL)
    ax.spines["polar"].set_color(BG_GRID)
    ax.grid(color=BG_GRID, linewidth=0.8)
    ax.set_title("Performance Radar\n(outer = better)", fontsize=9,
                 color=FG_TEXT, pad=14)

    for name in models:
        ax.plot([], [], color=PALETTE[name], lw=2, label=name)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28),
              ncol=2, fontsize=7, framealpha=0.4)


def plot_radar(results: dict, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 8), facecolor=BG_DARK,
                           subplot_kw=dict(polar=True))
    fig.suptitle("Multi-Metric Performance Radar",
                 fontsize=13, fontweight="bold", color=FG_TEXT, y=0.97)
    _draw_radar(ax, results, list(results.keys()))
    plt.tight_layout()
    _save(fig, "05_radar", dpi)


# ─────────────────────────────────────────────────────────────────────────────
#  6  COMPLEXITY vs RMSE SCATTER
# ─────────────────────────────────────────────────────────────────────────────
def plot_scatter(results: dict, dpi: int) -> None:
    models = list(results.keys())
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=BG_DARK)
    fig.suptitle("Model Complexity vs RMSE  —  Efficiency Frontier",
                 fontsize=13, fontweight="bold", color=FG_TEXT)

    params_k = [results[m]["n_params"] / 1000 for m in models]
    rmse_v   = [results[m]["metrics"]["RMSE"]  for m in models]

    for name, pk, rv in zip(models, params_k, rmse_v):
        c = PALETTE[name]
        ax.scatter(pk, rv, color=c, s=220, zorder=5,
                   edgecolors=BG_DARK, linewidths=1.2)
        ax.annotate(name, (pk, rv),
                    xytext=(8, 6), textcoords="offset points",
                    fontsize=9, color=c, fontweight="bold")

    # Efficiency frontier
    pairs = sorted(zip(params_k, rmse_v))
    frontier_x, frontier_y, frontier_active = [], [], []
    best_rmse = float("inf")
    for pk, rv in pairs:
        if rv < best_rmse:
            frontier_x.append(pk)
            frontier_y.append(rv)
            best_rmse = rv
    ax.step(frontier_x, frontier_y, where="post",
            color=ACCENT, lw=1.5, linestyle="--", alpha=0.5,
            label="Efficiency frontier", zorder=3)

    ax.set_xlabel("Parameters (thousands)", fontsize=10, color=FG_TEXT)
    ax.set_ylabel("RMSE  (lower = better)", fontsize=10, color=FG_TEXT)
    ax.set_facecolor(BG_PANEL)
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, "06_complexity_scatter", dpi)


# ─────────────────────────────────────────────────────────────────────────────
#  7  ATTENTION WEIGHTS HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
def plot_attention(results: dict, dpi: int) -> None:
    if "Attention LSTM" not in results or "attn_weights" not in results["Attention LSTM"]:
        print("  ⚠️   No attention weights found — skipping attention plot.")
        return

    weights = results["Attention LSTM"]["attn_weights"]  # (N, T)
    N_SHOW  = min(60, len(weights))
    w       = weights[:N_SHOW]                            # (60, 30)

    # Custom teal→gold colormap
    cmap = LinearSegmentedColormap.from_list(
        "attn", [BG_PANEL, "#00FFA3", "#FFD700"], N=256
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                             gridspec_kw={"width_ratios": [2, 1]},
                             facecolor=BG_DARK)
    fig.suptitle("Bahdanau Attention Weights  —  What the Model Focuses On",
                 fontsize=13, fontweight="bold", color=FG_TEXT)

    # Heatmap
    im = axes[0].imshow(w, aspect="auto", cmap=cmap, vmin=0)
    axes[0].set_xlabel("Input Time Step (0 = oldest,  29 = most recent)",
                       fontsize=9)
    axes[0].set_ylabel("Test Sample Index", fontsize=9)
    axes[0].set_title("Attention Weight Heatmap (60 test samples × 30 steps)",
                      fontsize=9, color=FG_TEXT)
    axes[0].set_facecolor(BG_PANEL)
    plt.colorbar(im, ax=axes[0], label="α weight", shrink=0.85)

    # Mean attention profile
    mean_w = w.mean(axis=0)
    c_prof = PALETTE["Attention LSTM"]
    axes[1].fill_between(range(SEQ_LEN), mean_w,
                         alpha=0.25, color=c_prof)
    _glow(axes[1], range(SEQ_LEN), mean_w, c_prof, lw=2.5)
    peak = int(np.argmax(mean_w))
    axes[1].axvline(peak, color="#FFD700", lw=1.5, linestyle="--")
    axes[1].text(peak + 0.5, mean_w.max() * 0.95,
                 f"Peak at t={peak}", fontsize=8, color="#FFD700")
    axes[1].set_title("Mean Attention Profile", fontsize=9, color=FG_TEXT)
    axes[1].set_xlabel("Time Step", fontsize=9)
    axes[1].set_ylabel("Mean α Weight", fontsize=9)
    axes[1].set_facecolor(BG_PANEL)

    plt.tight_layout()
    _save(fig, "07_attention_weights", dpi)


# ─────────────────────────────────────────────────────────────────────────────
#  8  SEQ2SEQ MULTI-STEP FORECAST SHOWCASE
# ─────────────────────────────────────────────────────────────────────────────
def plot_seq2seq(results: dict, dpi: int) -> None:
    if "Seq2Seq" not in results or "y_pred_steps" not in results["Seq2Seq"]:
        print("  ⚠️   No Seq2Seq step data found — skipping seq2seq plot.")
        return

    pred_steps = results["Seq2Seq"]["y_pred_steps"]  # (N, 7)
    true_steps = results["Seq2Seq"]["y_true_steps"]  # (N, 7)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), facecolor=BG_DARK)
    fig.suptitle(
        "Seq2Seq Encoder-Decoder  —  7-Day Multi-Step Forecast Showcase",
        fontsize=14, fontweight="bold", color=FG_TEXT
    )
    axes = axes.flatten()

    c = PALETTE["Seq2Seq"]
    for i, ax in enumerate(axes[:6]):
        idx  = i * (len(pred_steps) // 6)
        yt   = true_steps[idx]
        yp   = pred_steps[idx]
        days = range(1, PRED_LEN + 1)

        ax.fill_between(days, yt, alpha=0.15, color=c)
        _glow(ax, days, yt, FG_MUTED, lw=2.0, alpha_main=0.7, n_glow=0)
        _glow(ax, days, yp, c, lw=2.5)
        ax.scatter(days, yt, color=FG_MUTED, s=50, zorder=6, edgecolors=BG_DARK)
        ax.scatter(days, yp, color=c,        s=50, zorder=6, edgecolors=BG_DARK)

        mae_i = np.mean(np.abs(yt - yp))
        ax.set_title(f"Sample #{idx+1}  |  MAE {mae_i:.2f}°C",
                     fontsize=9, color=c)
        ax.set_xlabel("Forecast Step (day)", fontsize=8)
        ax.set_ylabel("Temperature (°C)", fontsize=8)
        ax.set_xticks(days)
        ax.set_facecolor(BG_PANEL)
        if i == 0:
            ax.plot([], [], color=FG_MUTED, lw=2, label="Actual")
            ax.plot([], [], color=c, lw=2, label="Predicted")
            ax.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, "08_seq2seq_forecast", dpi)


# ─────────────────────────────────────────────────────────────────────────────
#  9  VANISHING GRADIENT ILLUSTRATION
# ─────────────────────────────────────────────────────────────────────────────
def plot_gradient(dpi: int) -> None:
    T = 60
    t = np.arange(T)

    scenarios = [
        (0.50, "#FF6B6B", "Vanishing  (|W| = 0.5)",
         "Gradient → 0\nEarly steps forgotten"),
        (0.90, "#FFD700", "Near-stable  (|W| = 0.9)",
         "Slow decay —\nstill problematic"),
        (1.00, "#00FFA3", "Stable  (|W| = 1.0)",
         "Perfect gradient flow\n(orthogonal init)"),
        (1.10, "#00D4FF", "Exploding  (|W| = 1.1)",
         "Gradient → ∞\nTraining diverges"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 6), facecolor=BG_DARK)
    fig.suptitle(
        "Backpropagation Through Time — Gradient Flow Analysis",
        fontsize=13, fontweight="bold", color=FG_TEXT
    )

    for ax, (W, color, title, note) in zip(axes, scenarios):
        grad = np.array([abs(W) ** i for i in range(T)])
        ax.semilogy(t, grad, color=color, lw=2.5)
        ax.fill_between(t, grad, alpha=0.12, color=color)
        ax.set_title(title, fontsize=9, fontweight="bold", color=color)
        ax.set_xlabel("Time steps back (BPTT)", fontsize=8)
        ax.set_ylabel("Gradient magnitude", fontsize=8)
        ax.set_facecolor(BG_PANEL)
        ax.text(0.97, 0.97, note,
                transform=ax.transAxes, fontsize=8,
                color=color, ha="right", va="top",
                style="italic",
                bbox=dict(boxstyle="round,pad=0.3",
                          fc=BG_DARK, ec=color, alpha=0.7))
        # Annotation at T=50
        val50 = abs(W) ** 50
        ax.annotate(f"t=50: {val50:.1e}",
                    xy=(50, val50), xytext=(35, val50 * (100 if W < 1 else 0.01)),
                    fontsize=7, color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=1))

    plt.tight_layout()
    _save(fig, "09_vanishing_gradient", dpi)


# ─────────────────────────────────────────────────────────────────────────────
#  10  SUMMARY TABLE  (image)
# ─────────────────────────────────────────────────────────────────────────────
def plot_summary_table(results: dict, dpi: int) -> None:
    models  = list(results.keys())
    rows    = []
    for m in models:
        mt = results[m]["metrics"]
        rows.append([m,
                     f"{mt['MAE']:.4f}",
                     f"{mt['RMSE']:.4f}",
                     f"{mt['MAPE']:.2f}%",
                     f"{mt['R2']:.4f}",
                     f"{results[m]['n_params']:,}",
                     str(results[m]['epochs'])])

    # Sort by RMSE
    rows.sort(key=lambda r: float(r[2]))

    col_labels = ["Model", "MAE ↓", "RMSE ↓", "MAPE ↓", "R² ↑", "Params", "Epochs"]

    fig, ax = plt.subplots(figsize=(14, 4.5), facecolor=BG_DARK)
    ax.set_facecolor(BG_DARK)
    ax.axis("off")
    fig.suptitle("Model Comparison — Summary Table",
                 fontsize=13, fontweight="bold", color=FG_TEXT, y=1.02)

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.2)

    # Style header
    for j, _ in enumerate(col_labels):
        cell = tbl[(0, j)]
        cell.set_facecolor(BG_GRID)
        cell.set_text_props(color=ACCENT, fontweight="bold")

    # Style rows
    for i, row in enumerate(rows):
        model_name = row[0]
        rc = PALETTE.get(model_name, FG_TEXT)
        for j in range(len(col_labels)):
            cell = tbl[(i + 1, j)]
            cell.set_facecolor(BG_PANEL if i % 2 == 0 else BG_DARK)
            if j == 0:
                cell.set_text_props(color=rc, fontweight="bold")
            else:
                cell.set_text_props(color=FG_TEXT)

    plt.tight_layout()
    _save(fig, "10_summary_table", dpi)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
ALL_PLOTS = {
    "dashboard"  : lambda r, d: plot_dashboard(r, d),
    "metrics"    : lambda r, d: plot_metrics(r, d),
    "loss_curves": lambda r, d: plot_loss_curves(r, d),
    "predictions": lambda r, d: plot_predictions(r, d),
    "radar"      : lambda r, d: plot_radar(r, d),
    "scatter"    : lambda r, d: plot_scatter(r, d),
    "attention"  : lambda r, d: plot_attention(r, d),
    "seq2seq"    : lambda r, d: plot_seq2seq(r, d),
    "gradient"   : lambda _, d: plot_gradient(d),
    "table"      : lambda r, d: plot_summary_table(r, d),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate premium RNN plots")
    parser.add_argument("--plot", choices=list(ALL_PLOTS) + ["all"],
                        default="all", help="Which plot to generate")
    parser.add_argument("--dpi",  type=int, default=200,
                        help="Output DPI (default: 200)")
    args = parser.parse_args()

    _apply_style()
    results = _load()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"\n  Generating plots → ./{PLOTS_DIR}/\n")

    if args.plot == "all":
        for pname, fn in ALL_PLOTS.items():
            print(f"  [{pname}]")
            fn(results, args.dpi)
    else:
        ALL_PLOTS[args.plot](results, args.dpi)

    print(f"\n  All plots saved in ./{PLOTS_DIR}/")
