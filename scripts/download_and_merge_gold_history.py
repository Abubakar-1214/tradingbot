"""
Downloads historical XAUUSD H1 data from HuggingFace (2004-2025) and
merges with local MT5 broker data (up to October 2026).
"""

import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "xauusd_h1.csv"
HF_URL = "https://huggingface.co/datasets/ZombitX64/xauusd-gold-price-historical-data-2004-2025/resolve/main/XAU_1h_data.jsonl"


def fetch_huggingface_h1() -> pd.DataFrame:
    print(f"Downloading historical Gold H1 (2004-2025) from Hugging Face...")
    resp = requests.get(HF_URL, stream=True, timeout=60)
    resp.raise_for_status()

    records = []
    for line in resp.iter_lines(decode_unicode=True):
        if line:
            item = json.loads(line)
            # Example: {"Date":"2004.06.11 07:00","Open":384.0,"High":384.3,"Low":383.3,"Close":383.8,"Volume":44}
            dt_str = item.get("Date") or item.get("time")
            # Parse datetime: format 2004.06.11 07:00 or 2004-06-11 07:00:00
            dt = pd.to_datetime(dt_str.replace(".", "-"))
            records.append({
                "time": dt,
                "open": float(item["Open"]),
                "high": float(item["High"]),
                "low": float(item["Low"]),
                "close": float(item["Close"]),
                "volume": float(item.get("Volume", 0)),
            })

    df = pd.DataFrame(records)
    print(f"Downloaded {len(df):,} bars from Hugging Face ({df['time'].iloc[0]} to {df['time'].iloc[-1]})")
    return df


def merge_with_mt5_data(hf_df: pd.DataFrame, local_path: Path) -> pd.DataFrame:
    if local_path.exists():
        print(f"Loading local MT5 data from {local_path}...")
        local_df = pd.read_csv(local_path)
        local_df["time"] = pd.to_datetime(local_df["time"])
        print(f"Local MT5 data has {len(local_df):,} bars ({local_df['time'].iloc[0]} to {local_df['time'].iloc[-1]})")

        # Combine: Prioritize MT5 data for overlapping periods (post-2014) because it matches the live broker
        earliest_local = local_df["time"].min()
        hf_filtered = hf_df.loc[hf_df["time"] < earliest_local]
        print(f"Using {len(hf_filtered):,} pre-2014 bars from Hugging Face + {len(local_df):,} broker bars from MT5")

        combined = pd.concat([hf_filtered, local_df], ignore_index=True)
    else:
        combined = hf_df

    combined = combined.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    # Ensure correct column ordering
    combined = combined[["time", "open", "high", "low", "close", "volume"]]
    combined["time"] = combined["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return combined


def main():
    hf_df = fetch_huggingface_h1()
    merged = merge_with_mt5_data(hf_df, OUTPUT_FILE)

    # Save backup of current file if it exists
    if OUTPUT_FILE.exists():
        backup_path = OUTPUT_FILE.with_name("xauusd_h1_backup_premerge.csv")
        OUTPUT_FILE.replace(backup_path)
        print(f"Backed up previous data to {backup_path}")

    merged.to_csv(OUTPUT_FILE, index=False)
    print("\n" + "=" * 65)
    print(f" Successfully written expanded dataset to: {OUTPUT_FILE}")
    print(f" Total rows: {len(merged):,}")
    print(f" Start date: {merged['time'].iloc[0]}")
    print(f" End date:   {merged['time'].iloc[-1]}")
    print("=" * 65)


if __name__ == "__main__":
    main()
