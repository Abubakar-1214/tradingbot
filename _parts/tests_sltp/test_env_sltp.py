"""Subtask 1 verification: env sl_tp_action composite mode + backward compat.

- test_backward_compat_trace_identical: new env with sl_tp_action=False must
  produce a byte-identical step trace (obs, reward, done, info) and
  episode_stats to the ORIGINAL committed env under the same seed/actions.
- test_model_mode_episode_runs_and_counts: sl_tp_action=True episode runs,
  sl/tp_hits counted, forced closes happen, episode terminates.
- test_model_mode_done_on_sl_tp_hit: on a huge adverse bar the composite
  action's SL bucket triggers a forced close (position closed, sl_hits=1).
- test_composite_decode_roundtrip: every idx in [0,75) decodes to a valid
  (direction, sl_frac, tp_frac) with fracs from the bucket tuples.
- test_defaults_preserved: legacy 3/2-action space + decode unchanged.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(r"e:\Desktop\NeoMind\Bazz\autonoumuse_trader")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from env.dreamer_trading_env import (  # noqa: E402
    DEFAULT_ENV_KWARGS,
    EVAL_ENV_OVERRIDES,
    SL_BUCKETS,
    TP_BUCKETS,
    RealisticTradingEnv,
)

# ---------------------------------------------------------------------------
# Load the ORIGINAL committed env (git HEAD) for byte-identical comparison.
# ---------------------------------------------------------------------------
def _load_original_env():
    src = subprocess.run(
        ["git", "-C", str(REPO), "show", "HEAD:env/dreamer_trading_env.py"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    ns = {}
    exec(compile(src, "<orig_env>", "exec"), ns)
    return ns["RealisticTradingEnv"]


OrigEnv = _load_original_env()


def _make_data(n=1200, window=64, n_features=7, seed=1234):
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((n, n_features)).astype(np.float32)
    returns = (rng.standard_normal(n) * 0.001).astype(np.float32)
    timestamps = np.datetime64("2020-01-01", "h") + np.arange(n).astype(
        "timedelta64[h]"
    )
    return features, returns, timestamps, window


def _run_episode(env_cls, features, returns, timestamps, window, actions, seed=999):
    env = env_cls(
        features,
        returns,
        timestamps=timestamps,
        window=window,
        allow_short=True,
        seed=seed,
        max_episode_steps=600,
    )
    env.reset()
    trace = []
    for action in actions:
        onehot = np.zeros(env.action_space, dtype=np.float32)
        onehot[action] = 1.0
        obs, reward, done, info = env.step(onehot)
        trace.append((obs.copy(), float(reward), bool(done), dict(info)))
        if done:
            break
    stats = env.episode_stats()
    return trace, stats


def _actions_seq(n, action_space, seed=42):
    rng = np.random.default_rng(seed)
    return [int(rng.integers(0, action_space)) for _ in range(n)]


def test_backward_compat_trace_identical():
    features, returns, timestamps, window = _make_data()
    actions = _actions_seq(120, 3, seed=777)

    trace_orig, stats_orig = _run_episode(
        OrigEnv, features, returns, timestamps, window, actions, seed=999
    )
    trace_new, stats_new = _run_episode(
        RealisticTradingEnv, features, returns, timestamps, window, actions, seed=999
    )

    assert len(trace_new) == len(trace_orig), "trace lengths differ"
    for i, (o, n) in enumerate(zip(trace_orig, trace_new)):
        (obs_o, rew_o, done_o, info_o) = o
        (obs_n, rew_n, done_n, info_n) = n
        assert np.array_equal(obs_o, obs_n), f"obs differs at step {i}"
        assert rew_o == rew_n, f"reward differs at step {i}: {rew_o} vs {rew_n}"
        assert done_o == done_n, f"done differs at step {i}"
        for k in info_o:
            assert info_o[k] == info_n[k], (
                f"info[{k}] differs at step {i}: {info_o[k]} vs {info_n[k]}"
            )
    assert stats_new == stats_orig, f"episode_stats differ:\n{stats_orig}\n{stats_new}"


def test_model_mode_episode_runs_and_counts():
    features, returns, timestamps, window = _make_data()
    env = RealisticTradingEnv(
        features,
        returns,
        timestamps=timestamps,
        window=window,
        allow_short=True,
        seed=5,
        max_episode_steps=400,
        sl_tp_action=True,
    )
    assert env.action_space == 75
    env.reset()
    done = False
    steps = 0
    while not done and steps < 400:
        action = int(steps % 75)  # cycle all composite actions
        onehot = np.zeros(75, dtype=np.float32)
        onehot[action] = 1.0
        obs, reward, done, info = env.step(onehot)
        assert np.isfinite(reward), f"non-finite reward at step {steps}"
        steps += 1
    stats = env.episode_stats()
    # SL/TP forced closes (trades closed by a stop/target) must be counted by
    # the same counters used in legacy mode; at least one must have fired in
    # this deterministic run with 400 steps of direction cycling.
    assert env.sl_hits + env.tp_hits >= 1
    assert stats["sl_hits"] == env.sl_hits and stats["tp_hits"] == env.tp_hits
    assert stats["trades"] >= env.sl_hits + env.tp_hits
    # Episode must have terminated through a valid path (done, step cap, or
    # running out of data).
    assert done or steps == 400 or env.t >= env.T - 1


def test_model_mode_done_on_sl_tp_hit():
    # Force a stop-loss hit: big adverse bar move against a long, tight SL bucket.
    n, window = 300, 64
    rng = np.random.default_rng(0)
    features = rng.standard_normal((n, 5)).astype(np.float32)
    returns = np.zeros(n, dtype=np.float32)
    returns[window + 1] = -0.10  # single huge adverse move at bar window+1
    timestamps = np.datetime64("2021-01-01", "h") + np.arange(n).astype(
        "timedelta64[h]"
    )
    env = RealisticTradingEnv(
        features,
        returns,
        timestamps=timestamps,
        window=window,
        allow_short=True,
        seed=1,
        max_episode_steps=100,
        random_start=False,  # start exactly at t=window so steps hit returns[65]
        sl_tp_action=True,
    )
    env.reset()
    # composite idx 1*25 + 0*5 + 0 = 25 -> direction long, SL 0.005, TP 0.005
    onehot = np.zeros(75, dtype=np.float32)
    onehot[25] = 1.0
    # Step 1 at t=64: opens the long (return 0.0, no adverse move yet).
    obs, reward, done, info = env.step(onehot)
    assert info["forced_close"] is False
    assert env.pos == 1
    # Step 2 at t=65: return -0.10 -> cumulative trade move -0.10 <= -0.005 SL.
    obs, reward, done, info = env.step(onehot)
    assert info["forced_close"] is True, "expected SL forced close on big adverse move"
    assert env.sl_hits == 1 and env.tp_hits == 0
    assert env.pos == 0, "position must be flat after SL hit"


def test_composite_decode_roundtrip():
    features, returns, timestamps, window = _make_data(n=300)
    env = RealisticTradingEnv(
        features,
        returns,
        timestamps=timestamps,
        window=window,
        allow_short=True,
        seed=3,
        max_episode_steps=50,
        sl_tp_action=True,
    )
    for idx in range(75):
        onehot = np.zeros(75, dtype=np.float32)
        onehot[idx] = 1.0
        direction, sl_frac, tp_frac = env._decode_action(onehot)
        assert direction in (0, 1, -1)
        assert sl_frac in SL_BUCKETS, f"idx {idx} sl_frac {sl_frac} not in buckets"
        assert tp_frac in TP_BUCKETS, f"idx {idx} tp_frac {tp_frac} not in buckets"
        dir_idx = idx // 25
        sl_idx = (idx // 5) % 5
        tp_idx = idx % 5
        assert (dir_idx == 1) == (direction == 1), f"idx {idx} direction mismatch"
        assert sl_frac == SL_BUCKETS[sl_idx]
        assert tp_frac == TP_BUCKETS[tp_idx]

    # Long-only env: only dir_idx 1 maps to long (1), others flat (0).
    env_lo = RealisticTradingEnv(
        features,
        returns,
        timestamps=timestamps,
        window=window,
        allow_short=False,
        seed=3,
        max_episode_steps=50,
        sl_tp_action=True,
    )
    assert env_lo.action_space == 75
    for idx in (0, 2, 50, 74):
        onehot = np.zeros(75, dtype=np.float32)
        onehot[idx] = 1.0
        direction, sl_frac, tp_frac = env_lo._decode_action(onehot)
        assert direction == (1 if idx // 25 == 1 else 0)
        assert sl_frac in SL_BUCKETS and tp_frac in TP_BUCKETS


def test_defaults_preserved():
    assert DEFAULT_ENV_KWARGS["sl_tp_action"] is False
    assert "sl_tp_action" not in EVAL_ENV_OVERRIDES
    # Legacy 3-action space unaffected when sl_tp_action=False
    features, returns, timestamps, window = _make_data(n=300)
    env = RealisticTradingEnv(
        features, returns, timestamps=timestamps, window=window, seed=1
    )
    assert env.action_space == 3
    assert env._decode_action(np.array([0.0, 1.0, 0.0])) == 1
    env_lo = RealisticTradingEnv(
        features,
        returns,
        timestamps=timestamps,
        window=window,
        allow_short=False,
        seed=1,
    )
    assert env_lo.action_space == 2
    assert env_lo._decode_action(np.array([0.0, 1.0])) == 1
    assert env_lo._decode_action(np.array([1.0, 0.0])) == 0


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
