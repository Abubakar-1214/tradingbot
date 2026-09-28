"""
scripts/_apply_feature_fixes.py — one-shot targeted replacements for P1 fixes.

Applies surgical text replacements (verified by exact-match assertions) to:
  1. features/timeframe_features.py — handle BOTH 'volume' and 'tick_volume'
     (MT5 supplies tick_volume; CSVs supply volume).  P1.
  2. features/calendar_features.py — vectorized O(N log E) implementation
     replacing the O(N*E) Python loop.  P1.

Safe to re-run (idempotent: skips already-applied replacements).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def apply(path: Path, old: str, new: str, label: str) -> bool:
    if not path.exists():
        print(f"[MISS-FILE] {label}: {path} not found")
        return False
    text = path.read_text(encoding="utf-8")
    if old not in text:
        print(f"[ALREADY-OR-SKIP] {label}: pattern not found (already applied?)")
        return False
    text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")
    print(f"[OK] {label}: replacement applied to {path.name}")
    return True


def main() -> int:
    ok = True

    # ------------------------------------------------------------------ #
    # 1. timeframe_features.py — volume/tick_volume
    # ------------------------------------------------------------------ #
    tf_path = REPO / "features" / "timeframe_features.py"

    old_vol = """    # 14. Volume ratio (current / 20-period average)
    avg_volume = df['volume'].rolling(20).mean()
    result[f'{tf_name}_volume_ratio'] = df['volume'] / avg_volume"""
    new_vol = """    # 14. Volume ratio (current / 20-period average)
    # MT5 supplies tick_volume, CSVs supply volume - handle BOTH (P1 fix).
    vol_col = 'volume' if 'volume' in df.columns else ('tick_volume' if 'tick_volume' in df.columns else None)
    if vol_col is not None:
        avg_volume = df[vol_col].rolling(20).mean()
        result[f'{tf_name}_volume_ratio'] = df[vol_col] / avg_volume
    else:
        result[f'{tf_name}_volume_ratio'] = 1.0"""
    ok &= apply(tf_path, old_vol, new_vol, "timeframe_features volume/tick_volume")

    # load_timeframe_data required-columns check must accept either.
    old_req = """    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")"""
    new_req = """    required = ['open', 'high', 'low', 'close']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    # Accept EITHER 'volume' (CSV) or 'tick_volume' (MT5) - P1 fix.
    if 'volume' not in df.columns and 'tick_volume' not in df.columns:
        raise ValueError("Missing volume column: need 'volume' or 'tick_volume'")"""
    ok &= apply(tf_path, old_req, new_req, "timeframe_features required cols")

    # ------------------------------------------------------------------ #
    # 2. calendar_features.py — vectorize the O(N*E) loop (P1)
    # ------------------------------------------------------------------ #
    cal_path = REPO / "features" / "calendar_features.py"

    old_loop_start = """    logger.info(f"Processing {len(df_timestamps):,} timestamps...")

    # Initialize result arrays
    hours_to_event = []
    days_since_event = []
    event_density = []
    is_high_impact = []
    in_event_window = []
    event_volatility_expected = []
    event_type_nfp = []
    event_type_fomc = []

    # Process each timestamp
    for i, ts in enumerate(df_timestamps):
        if i % 10000 == 0:
            logger.info(f"   Processing: {i:,} / {len(df_timestamps):,}")

        # Find next event
        next_event = find_next_event(ts, calendar)

        if next_event:
            # Feature 1: Hours to next event
            time_diff = (next_event['time'] - ts).total_seconds() / 3600.0
            hours_to_event.append(min(time_diff, 168.0))  # Cap at 1 week

            # Feature 4: Is high impact
            is_high = 1.0 if next_event.get('impact', 'MEDIUM') == 'HIGH' else 0.0
            is_high_impact.append(is_high)

            # Feature 5: In event window (±2 hours)
            in_window = 1.0 if abs(time_diff) <= 2.0 else 0.0
            in_event_window.append(in_window)

            # Feature 6: Expected volatility multiplier
            if next_event.get('impact', 'MEDIUM') == 'HIGH':
                vol_mult = 2.0
            elif next_event.get('impact', 'MEDIUM') == 'MEDIUM':
                vol_mult = 1.5
            else:
                vol_mult = 1.0
            event_volatility_expected.append(vol_mult)

            # Feature 7-8: Event types
            event_name = next_event.get('event', '').upper()
            is_nfp = 1.0 if 'NFP' in event_name or 'NONFARM' in event_name else 0.0
            is_fomc = 1.0 if 'FOMC' in event_name or 'FEDERAL RESERVE' in event_name else 0.0
            event_type_nfp.append(is_nfp)
            event_type_fomc.append(is_fomc)
        else:
            # No future events
            hours_to_event.append(168.0)
            is_high_impact.append(0.0)
            in_event_window.append(0.0)
            event_volatility_expected.append(1.0)
            event_type_nfp.append(0.0)
            event_type_fomc.append(0.0)

        # Find last event
        last_event = find_last_event(ts, calendar)

        if last_event:
            # Feature 2: Days since last event
            time_since = (ts - last_event['time']).total_seconds() / 86400.0
            days_since_event.append(min(time_since, 30.0))  # Cap at 30 days
        else:
            days_since_event.append(30.0)

        # Feature 3: Event density (upcoming events in next 7 days)
        density = count_upcoming_events(ts, calendar, days=7)
        event_density.append(min(density, 10.0))  # Cap at 10

    # Assign to result DataFrame
    result['hours_to_event'] = hours_to_event
    result['days_since_event'] = days_since_event
    result['event_density'] = event_density
    result['is_high_impact'] = is_high_impact
    result['in_event_window'] = in_event_window
    result['event_volatility_expected'] = event_volatility_expected
    result['event_type_nfp'] = event_type_nfp
    result['event_type_fomc'] = event_type_fomc"""
    new_loop = """    logger.info(f"Processing {len(df_timestamps):,} timestamps (vectorized)...")

    # --- Build sorted numpy arrays of event times for O(N log E) searchsorted ---
    times = np.asarray(pd.to_datetime([e['time'] for e in calendar], utc=True).values, dtype='datetime64[ns]')
    impacts = np.array([str(e.get('impact', 'MEDIUM')).upper() for e in calendar])
    names = np.array([str(e.get('event', '')).upper() for e in calendar])

    ts_arr = np.asarray(pd.to_datetime(df_timestamps, utc=True).values, dtype='datetime64[ns]')

    # For each timestamp: next event = first event strictly AFTER it.
    next_idx = np.searchsorted(times, ts_arr, side='right')
    # Last event = last event at-or-before it.
    last_idx = np.searchsorted(times, ts_arr, side='right') - 1

    n = len(ts_arr)
    hours_to_event = np.full(n, 168.0)
    is_high_impact = np.zeros(n)
    in_event_window = np.zeros(n)
    event_volatility_expected = np.ones(n)
    event_type_nfp = np.zeros(n)
    event_type_fomc = np.zeros(n)
    days_since_event = np.full(n, 30.0)
    event_density = np.zeros(n)

    mask_next = next_idx < len(times)
    if mask_next.any():
        next_ts = times[next_idx[mask_next]]
        td_hours = (next_ts - ts_arr[mask_next]).astype('float64') / 3.6e12  # ns -> hours
        hours_to_event[mask_next] = np.minimum(td_hours, 168.0)
        imp = impacts[next_idx[mask_next]]
        is_high_impact[mask_next] = (imp == 'HIGH').astype(float)
        in_event_window[mask_next] = (np.abs(td_hours) <= 2.0).astype(float)
        ev = np.full(len(imp), 1.0)
        ev[imp == 'HIGH'] = 2.0
        ev[imp == 'MEDIUM'] = 1.5
        event_volatility_expected[mask_next] = ev
        en = names[next_idx[mask_next]]
        event_type_nfp[mask_next] = np.char.find(en, 'NFP') >= 0
        event_type_nfp[mask_next] |= np.char.find(en, 'NONFARM') >= 0
        event_type_fomc[mask_next] = np.char.find(en, 'FOMC') >= 0
        event_type_fomc[mask_next] |= np.char.find(en, 'FEDERAL RESERVE') >= 0

    mask_last = last_idx >= 0
    if mask_last.any():
        last_ts = times[last_idx[mask_last]]
        td_days = (ts_arr[mask_last] - last_ts).astype('float64') / 8.64e13  # ns -> days
        days_since_event[mask_last] = np.minimum(td_days, 30.0)

    # Event density: count events in (ts, ts + 7 days] via two searchsorteds.
    future_7d = ts_arr + np.timedelta64(7, 'D')
    right = np.searchsorted(times, future_7d, side='right')
    event_density = np.minimum((right - next_idx).astype(float), 10.0)

    # Assign to result DataFrame
    result['hours_to_event'] = hours_to_event
    result['days_since_event'] = days_since_event
    result['event_density'] = event_density
    result['is_high_impact'] = is_high_impact
    result['in_event_window'] = in_event_window
    result['event_volatility_expected'] = event_volatility_expected
    result['event_type_nfp'] = event_type_nfp
    result['event_type_fomc'] = event_type_fomc"""
    ok &= apply(cal_path, old_loop_start, new_loop, "calendar_features vectorized")

    print()
    print("ALL_OK" if ok else "SOME_SKIPPED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
