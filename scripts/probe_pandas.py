"""scripts/probe_pandas.py — verify pandas 3.0.5 resample label/closed semantics."""
import pandas as pd
import numpy as np

idx = pd.date_range("2020-01-01", periods=8, freq="h")
df = pd.DataFrame({"close": np.arange(8.0), "open": np.arange(8.0)}, index=idx)

# Standard causal resample: label='right', closed='right'
r = df.resample("4h", label="right", closed="right").agg(
    {"open": "first", "close": "last"}
)
print("LABEL_RIGHT_CLOSED_RIGHT:")
print(r.index.tolist())
print(r["close"].tolist())

# Shift by one higher-TF period -> leak-free (value only known after candle closes)
r_shift = r.shift(1)
print("SHIFTED:")
print(r_shift["close"].tolist())

# ffill onto base index (causal merge)
merged = r_shift.reindex(df.index, method="ffill")
print("FFILL_ON_BASE:")
print(merged["close"].tolist())

# pct_change / rolling still fine
print("PCT_CHANGE:", df["close"].pct_change().iloc[-1])

# Check that rolling windows don't look ahead (min_periods default == window)
print("ROLLING_OK", df["close"].rolling(3).mean().iloc[-1])

print("PANDAS", pd.__version__)
