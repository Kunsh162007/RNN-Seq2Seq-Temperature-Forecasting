"""
download_data.py
────────────────
Downloads the Daily Minimum Temperatures dataset (Melbourne, 1981-1990)
and saves it to the data/ directory.

Run:
    python download_data.py
"""

import os
import requests
import pandas as pd
import numpy as np

DATA_DIR  = "data"
DATA_FILE = os.path.join(DATA_DIR, "daily_min_temp.csv")
URL       = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"

os.makedirs(DATA_DIR, exist_ok=True)

# ── Try downloading real dataset ───────────────────────────────────────────────
print("Downloading dataset...")
try:
    r = requests.get(URL, timeout=15)
    r.raise_for_status()
    with open(DATA_FILE, "wb") as f:
        f.write(r.content)
    df = pd.read_csv(DATA_FILE, parse_dates=["Date"], index_col="Date")
    df.columns = ["Temp"]
    df.to_csv(DATA_FILE)
    print(f"✅  Dataset saved → {DATA_FILE}")
    print(f"    Shape      : {df.shape}")
    print(f"    Date range : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"    Temp range : {df.Temp.min():.1f}°C – {df.Temp.max():.1f}°C")
except Exception as e:
    print(f"⚠️   Network error ({e})")
    print("    Generating synthetic equivalent dataset instead...")
    rng = np.random.default_rng(42)
    dates = pd.date_range("1981-01-01", periods=3650, freq="D")
    t = np.arange(3650)
    temp = (
        10.0
        + 8.0 * np.sin(2 * np.pi * t / 365.25)   # annual cycle
        + 0.002 * t                                  # slow warming trend
        + rng.normal(0, 1.5, 3650)                   # day-to-day noise
    )
    df = pd.DataFrame({"Temp": temp}, index=dates)
    df.index.name = "Date"
    df.to_csv(DATA_FILE)
    print(f"✅  Synthetic dataset saved → {DATA_FILE}  ({len(df)} samples)")
