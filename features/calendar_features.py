"""
Economic Calendar Advanced Features Module

Computes 8 features from economic events:
- Event Timing (3): hours to event, days since event, event density
- Event Impact (3): is high impact, in event window, expected volatility
- Event Type (2): NFP detection, FOMC detection

These features make the AI aware of major economic releases and their impact.
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_economic_calendar(filepath='data/economic_events_2015_2025.json'):
    """
    Load economic calendar from JSON file

    Returns:
        List of event dicts with keys: time, event, impact
    """
    logger.info(f"📅 Loading economic calendar from {filepath}...")

    filepath = Path(filepath)

    if not filepath.exists():
        logger.warning(f"⚠️  Calendar file not found: {filepath}")
        return []

    with open(filepath, 'r') as f:
        events = json.load(f)

    # Convert datetime strings to datetime objects and rename to 'time'
    for event in events:
        event['time'] = pd.to_datetime(event['datetime'])
        # Keep datetime for backward compatibility if needed
        if 'datetime' in event and 'time' not in event:
            event['time'] = event['datetime']

    logger.info(f"   ✅ Loaded {len(events)} economic events")

    return events


def find_next_event(timestamp, events):
    """
    Find the next economic event after given timestamp

    Args:
        timestamp: Current time
        events: List of event dicts

    Returns:
        Dict with next event info, or None if no future events
    """
    future_events = [e for e in events if e['time'] > timestamp]

    if not future_events:
        return None

    # Return the nearest future event
    return min(future_events, key=lambda e: e['time'])


def find_last_event(timestamp, events):
    """
    Find the most recent economic event before given timestamp

    Args:
        timestamp: Current time
        events: List of event dicts

    Returns:
        Dict with last event info, or None if no past events
    """
    past_events = [e for e in events if e['time'] <= timestamp]

    if not past_events:
        return None

    # Return the most recent past event
    return max(past_events, key=lambda e: e['time'])


def count_upcoming_events(timestamp, events, days=7):
    """
    Count events in the next N days

    Args:
        timestamp: Current time
        events: List of event dicts
        days: Number of days to look ahead

    Returns:
        Count of upcoming events
    """
    future_time = timestamp + timedelta(days=days)
    upcoming = [e for e in events if timestamp < e['time'] <= future_time]

    return len(upcoming)


def compute_calendar_features(df_timestamps, calendar):
    """
    Compute 8 calendar-based features

    Args:
        df_timestamps: DataFrame with DatetimeIndex (from gold data)
        calendar: List of event dicts from load_economic_calendar()

    Returns:
        DataFrame with 8 calendar features
    """
    logger.info("="*70)
    logger.info("📅 COMPUTING ECONOMIC CALENDAR FEATURES")
    logger.info("="*70)

    result = pd.DataFrame(index=df_timestamps)

    if not calendar:
        logger.warning("⚠️  No calendar data available, filling with zeros")
        result['hours_to_event'] = 168.0  # 1 week default
        result['days_since_event'] = 7.0
        result['event_density'] = 0.0
        result['is_high_impact'] = 0.0
        result['in_event_window'] = 0.0
        result['event_volatility_expected'] = 1.0
        result['event_type_nfp'] = 0.0
        result['event_type_fomc'] = 0.0
        return result

    logger.info(f"Processing {len(df_timestamps):,} timestamps (vectorized)...")

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
    result['event_type_fomc'] = event_type_fomc

    # Normalize some features
    result['hours_to_event'] = result['hours_to_event'] / 168.0  # Normalize to 0-1
    result['days_since_event'] = result['days_since_event'] / 30.0
    result['event_density'] = result['event_density'] / 10.0

    # Fill any NaNs
    result = result.fillna(0.0)

    # Summary
    logger.info("\n" + "="*70)
    logger.info("✅ CALENDAR FEATURES COMPLETE")
    logger.info("="*70)
    logger.info(f"✅ Generated {result.shape[1]} calendar features")
    logger.info(f"✅ Processed {len(result):,} timestamps")

    # Statistics
    high_impact_count = result['is_high_impact'].sum()
    event_window_count = result['in_event_window'].sum()

    logger.info(f"\n📊 Calendar statistics:")
    logger.info(f"   • High impact events ahead: {int(high_impact_count):,} timestamps")
    logger.info(f"   • In event window (±2h): {int(event_window_count):,} timestamps")

    # List features
    logger.info("\n📊 Features created:")
    for col in result.columns:
        logger.info(f"   • {col}")

    return result


def test_calendar_features():
    """
    Test function to verify calendar features work correctly
    """
    logger.info("\n" + "="*70)
    logger.info("🧪 TESTING CALENDAR FEATURES")
    logger.info("="*70)

    try:
        # Load calendar
        logger.info("\n1️⃣ Loading economic calendar...")
        calendar = load_economic_calendar()

        # Load gold data for timestamps
        logger.info("\n2️⃣ Loading gold data for timestamps...")
        df_gold = pd.read_csv('data/xauusd_m5.csv')
        df_gold['time'] = pd.to_datetime(df_gold['time'])
        df_gold = df_gold.set_index('time').sort_index()

        # Take a subset for testing (first 10k bars)
        df_gold_subset = df_gold.head(10000)

        logger.info("\n3️⃣ Computing calendar features...")
        calendar_features = compute_calendar_features(df_gold_subset.index, calendar)

        logger.info("\n✅ Calendar features computed successfully!")

        # Check for NaNs
        nan_count = calendar_features.isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"⚠️  {nan_count} NaN values found")
        else:
            logger.info("✅ No NaN values")

        # Show sample
        logger.info("\n📊 Sample data:")
        logger.info(calendar_features.head(10))

        return calendar_features

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    # Run test
    calendar_feat = test_calendar_features()

    logger.info("\n" + "="*70)
    logger.info("✅ CALENDAR FEATURES MODULE READY")
    logger.info("="*70)

    logger.info("""
📋 USAGE:
    from features.calendar_features import load_economic_calendar, compute_calendar_features

    # Load calendar
    calendar = load_economic_calendar()

    # Compute features (pass DataFrame index)
    calendar_features = compute_calendar_features(df.index, calendar)
    """)
