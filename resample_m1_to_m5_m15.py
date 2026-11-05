"""
resample_m1_to_m5_m15.py

Resamples the full 3.32 Million rows from data/xauusd_m1.csv
into clean, validated M5 and M15 datasets in data/new_data/
spanning 2014 to 2026 (12+ full years).
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
NEW_DATA_DIR = DATA_DIR / "new_data"

def main():
    m1_path = DATA_DIR / "xauusd_m1.csv"
    if not m1_path.exists():
        logger.error(f"Cannot find {m1_path}")
        return

    logger.info(f"Loading 3.32M rows from {m1_path} (this may take ~10-15 seconds)...")
    df = pd.read_csv(m1_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
    logger.info(f"Loaded {len(df):,} clean M1 rows from {df['time'].iloc[0]} to {df['time'].iloc[-1]}")

    df_indexed = df.set_index("time")

    # Resample M5
    logger.info("Resampling to 5-Minute (M5) bars with right-closed alignment...")
    df_m5 = df_indexed.resample("5min", closed="right", label="right").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna().reset_index()
    m5_out = NEW_DATA_DIR / "xauusd_m5.csv"
    df_m5.to_csv(m5_out, index=False)
    logger.info(f"✓ Saved {m5_out} ({len(df_m5):,} rows | {df_m5['time'].iloc[0]} to {df_m5['time'].iloc[-1]})")

    # Resample M15
    logger.info("Resampling to 15-Minute (M15) bars with right-closed alignment...")
    df_m15 = df_indexed.resample("15min", closed="right", label="right").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna().reset_index()
    m15_out = NEW_DATA_DIR / "xauusd_m15.csv"
    df_m15.to_csv(m15_out, index=False)
    logger.info(f"✓ Saved {m15_out} ({len(df_m15):,} rows | {df_m15['time'].iloc[0]} to {df_m15['time'].iloc[-1]})")

    # Update manifest
    manifest_path = NEW_DATA_DIR / "manifest.json"
    if manifest_path.exists():
        import json
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["datasets"]["xauusd_m5"] = {
            "rows": len(df_m5),
            "start": str(df_m5["time"].iloc[0]),
            "end": str(df_m5["time"].iloc[-1]),
            "file": "xauusd_m5.csv"
        }
        manifest["datasets"]["xauusd_m15"] = {
            "rows": len(df_m15),
            "start": str(df_m15["time"].iloc[0]),
            "end": str(df_m15["time"].iloc[-1]),
            "file": "xauusd_m15.csv"
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info("✓ Updated manifest.json")

    print("\n🎉 Resampling of M5 and M15 complete!")

if __name__ == "__main__":
    main()
