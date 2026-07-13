"""Fetch and cache daily OHLC data for all indices used in the project.

The notebooks read the cached CSVs rather than calling yfinance directly, so that they are
reproducible and do not break when the yfinance API changes (it now returns MultiIndex columns,
which silently breaks a lot of older notebooks).

Existing files are NOT overwritten: Yahoo occasionally revises historical bars, and re-downloading
would silently desynchronise the cached data from the executed notebook outputs.

    python scripts/fetch_data.py
"""

import os

import pandas as pd
import yfinance as yf

START = "2010-01-01"
END = "2024-01-01"

INDICES = {
    "gspc": "^GSPC",     # S&P 500          (parts 1-3)
    "dji": "^DJI",       # Dow Jones        (part 3 robustness)
    "ixic": "^IXIC",     # NASDAQ Composite (part 3 robustness)
    "gdaxi": "^GDAXI",   # DAX              (part 3 robustness)
    "ftse": "^FTSE",     # FTSE 100         (part 3 robustness)
    "n225": "^N225",     # Nikkei 225       (part 3 robustness)
}

for key, ticker in INDICES.items():
    out = f"data/{key}.csv"
    if os.path.exists(out):
        print(f"{out}: already cached, skipping (delete the file to force a re-download)")
        continue

    df = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close"]].dropna()
    df.index.name = "Date"
    df.to_csv(out)
    print(f"{out}: {len(df)} rows, {df.index[0].date()} → {df.index[-1].date()}")
