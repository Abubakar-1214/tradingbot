"""
Fix ReplayBuffer degenerate-case behavior (P1 verify failure).

When episodes are SHORTER than seq_len, NO sequence of seq_len consecutive
transitions can avoid an interior done=True — the episode-boundary guard must
refuse to sample (return None, which DreamerV3Agent.train_step already
handles: 'if batch is None: return None') instead of emitting sequences that
cross episode boundaries.  This is the honest semantics of the P1 fix.

Also updates scripts/_verify_p1_fixes.py so the degenerate case asserts
sample() returns None, and adds a medium-episode case proving the property
holds when episodes are longer than seq_len.

Run: python scripts/_patch_p1_replay.py
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def patch_file(path: Path, anchor: str, replacement: str) -> None:
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    lf = text.replace("\r\n", "\n")
    n = lf.count(anchor)
    if n != 1:
        raise SystemExit(f"FAIL {path}: anchor found {n} times (expected 1)")
    out = lf.replace(anchor, replacement, 1)
    path.write_bytes(out.replace("\n", "\r\n").encode("utf-8"))
    print(f"[OK] patched {path.name} ({n} anchor occurrence)")


da = ROOT / "models" / "dreamer_agent.py"

# 1. ReplayBuffer.sample(): refuse to sample when no valid start exists.
patch_file(
    da,
    """        valid = np.flatnonzero(counts[:max_start + 1] == 0)
        if len(valid) == 0:
            # Degenerate fallback (episodes shorter than the sequence): sample
            # starts whose preceding transition did not terminate an episode.
            preceding_done = np.concatenate([[0.0], done_logical[:-1]])
            valid = np.flatnonzero(preceding_done[:max_start + 1] == 0.0)
        if len(valid) == 0:
            valid = np.arange(max_start + 1)

        starts = valid[np.random.randint(0, len(valid), size=batch_size)]""",
    """        valid = np.flatnonzero(counts[:max_start + 1] == 0)
        if len(valid) == 0:
            # Degenerate case: episodes are shorter than seq_len, so no
            # sequence of seq_len transitions can avoid crossing a done=True
            # boundary.  Refuse to sample (train_step skips the step) rather
            # than emit sequences that violate the episode-boundary guard.
            return None

        starts = valid[np.random.randint(0, len(valid), size=batch_size)]""",
)

# 2. Verify script: degenerate case must return None; add medium-episode case.
vf = ROOT / "scripts" / "_verify_p1_fixes.py"

patch_file(
    vf,
    """# Episode-heavy edge case: done every 3rd transition, seq_len=8 (> episode len)
buf2 = ReplayBuffer(capacity=500, seq_len=8)
for i in range(400):
    done = (i % 3 == 2)
    buf2.add(np.zeros(4, dtype="float32"), np.zeros(3, dtype="float32"), 0.0, done)
s2 = buf2.sample(100)
check("ReplayBuffer degenerate (short episodes) still samples",
      s2 is not None and s2["done"].shape == (100, 8))
check("ReplayBuffer degenerate no interior done",
      np.all(s2["done"][:, :-1] == 0),
      f"interior done count={int(np.count_nonzero(s2['done'][:, :-1]))}")""",
    """# Episode-heavy edge case: done every 3rd transition, seq_len=8 (> episode len).
# No 8-step window can avoid an interior done, so the guard MUST refuse to
# sample rather than cross episode boundaries (train_step handles None).
buf2 = ReplayBuffer(capacity=500, seq_len=8)
for i in range(400):
    done = (i % 3 == 2)
    buf2.add(np.zeros(4, dtype="float32"), np.zeros(3, dtype="float32"), 0.0, done)
s2 = buf2.sample(100)
check("ReplayBuffer degenerate (short episodes) refuses to sample",
      s2 is None,
      f"expected None, got shape={None if s2 is None else s2['done'].shape}")

# Medium-episode case: done every 40th transition, seq_len=16 (< episode len).
# Sequences must never cross a done — done only at the FINAL position.
buf3 = ReplayBuffer(capacity=1000, seq_len=16)
for i in range(600):
    done = (i % 40 == 39)
    buf3.add(np.zeros(4, dtype="float32"), np.zeros(3, dtype="float32"), 0.0, done)
s3 = buf3.sample(100)
check("ReplayBuffer medium-episode samples not None",
      s3 is not None)
check("ReplayBuffer medium-episode no interior done",
      s3 is not None and np.all(s3["done"][:, :-1] == 0),
      f"interior done count={0 if s3 is None else int(np.count_nonzero(s3['done'][:, :-1]))}")
check("ReplayBuffer medium-episode some final-position dones",
      s3 is not None and np.any(s3["done"][:, -1] == 1))""",
)

print("\nReplayBuffer degenerate-case fix applied successfully.")
