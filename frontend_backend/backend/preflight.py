from __future__ import annotations

import csv
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path


@lru_cache(maxsize=64)
def csv_time_bounds(path_text: str, modified_ns: int, size: int) -> tuple[datetime | None, datetime | None]:
    """Cache CSV endpoint timestamps by file metadata; never reparse each UI poll."""
    del modified_ns, size
    path = Path(path_text)
    first = last = None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"time", "open", "high", "low", "close"}.issubset(reader.fieldnames or []):
            return None, None
        for row in reader:
            try:
                stamp = datetime.fromisoformat(row["time"].strip())
            except (ValueError, AttributeError):
                continue
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=timezone.utc)
            first = first or stamp
            last = stamp
    return first, last
