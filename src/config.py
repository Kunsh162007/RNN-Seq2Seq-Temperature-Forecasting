"""
src/config.py
─────────────
Central configuration — change hyperparameters here only.
"""

# ── Data ───────────────────────────────────────────────────────────────────────
DATA_FILE   = "data/daily_min_temp.csv"
SEQ_LEN     = 30       # input lookback window (days)
PRED_LEN    = 7        # Seq2Seq output horizon (days)
TEST_RATIO  = 0.15
VAL_RATIO   = 0.10

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE  = 64
EPOCHS      = 100
PATIENCE    = 12       # early stopping
LR          = 1e-3
SEED        = 42

# ── Model sizes ────────────────────────────────────────────────────────────────
UNITS       = 64       # base hidden size
DROPOUT     = 0.2

# ── Paths ──────────────────────────────────────────────────────────────────────
MODELS_DIR  = "models"
RESULTS_DIR = "results"
PLOTS_DIR   = "plots"

# ── Model registry ─────────────────────────────────────────────────────────────
# Keys used consistently across train.py / visualize.py
MODEL_NAMES = [
    "Vanilla RNN",
    "LSTM",
    "GRU",
    "Stacked LSTM",
    "Bidirectional LSTM",
    "Attention LSTM",
    "Seq2Seq",
]

# Premium colour palette (used in visualize.py)
PALETTE = {
    "Vanilla RNN"        : "#FF6B6B",   # coral red
    "LSTM"               : "#00D4FF",   # electric cyan
    "GRU"                : "#00FFA3",   # mint green
    "Stacked LSTM"       : "#FFD700",   # gold
    "Bidirectional LSTM" : "#FF9F43",   # amber
    "Attention LSTM"     : "#A29BFE",   # soft violet
    "Seq2Seq"            : "#FD79A8",   # rose
}

BG_DARK     = "#0D1117"   # GitHub-dark background
BG_PANEL    = "#161B22"   # panel background
BG_GRID     = "#21262D"   # grid lines
FG_TEXT     = "#E6EDF3"   # primary text
FG_MUTED    = "#7D8590"   # secondary text
ACCENT      = "#00D4FF"   # highlight accent
