"""Smoke tests for RealisticTradingEnv (run: python -m pytest env/test_dreamer_trading_env.py)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.dreamer_trading_env import ACCOUNT_STATE_DIM, RealisticTradingEnv  # noqa: E402

T, F, W = 3000, 4, 16


def _data(seed=0, scale=0.001):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(T, F)).astype(np.float32)
    r = (rng.normal(size=T) * scale).astype(np.float32)
    ts = np.datetime64("2020-01-01T00:00") + np.arange(T) * np.timedelta64(1, "h")
    return X, r, ts


def _env(**kw):
    X, r, ts = _data()
    base = dict(window=W, slippage=0.0, spread=0.0, commission=0.0, swap_long=0.0,
                swap_short=0.0, stop_loss=None, max_drawdown=None, random_start=False,
                max_episode_steps=None, reward_scale=1.0, seed=0)
    base.update(kw)
    return RealisticTradingEnv(X, r, timestamps=ts, **base), X, r


def onehot(i, n=3):
    a = np.zeros(n, dtype=np.float32)
    a[i] = 1.0
    return a


def test_obs_shape_and_causality():
    env, X, r = _env()
    obs = env.reset()
    assert obs.shape == (W * F + ACCOUNT_STATE_DIM,)
    assert env.observation_space == obs.shape[0]
    np.testing.assert_array_equal(obs[:W * F], X[:W].reshape(-1))
    obs2, _, _, _ = env.step(onehot(1))
    # after one step the window ends at bar t=W (bar W's features), never beyond
    np.testing.assert_array_equal(obs2[:W * F], X[1:W + 1].reshape(-1))


def test_frictionless_pnl_matches_returns():
    env, X, r = _env()
    env.reset()
    _, reward_long, _, info = env.step(onehot(1))
    assert np.isclose(info["pnl"], r[W])
    assert np.isclose(reward_long, np.log1p(r[W]))
    _, reward_short, _, info = env.step(onehot(2))
    # flip long -> short: earns -r
    assert np.isclose(info["pnl"], -r[W + 1])
    assert info["position"] == -1


def test_costs_are_charged_on_position_changes_only():
    env, _, r = _env(spread=0.0004, commission=0.0001)
    env.reset()
    _, _, _, info = env.step(onehot(1))
    assert np.isclose(info["cost"], 0.0002 + 0.0001)      # half spread + commission
    _, _, _, info = env.step(onehot(1))
    assert info["cost"] == 0.0                              # holding is free
    _, _, _, info = env.step(onehot(2))
    assert np.isclose(info["cost"], 2 * (0.0002 + 0.0001))  # flip = 2 units traded
    assert env.n_trades == 1


def test_slippage_is_mostly_adverse_and_vol_scaled():
    env, _, _ = _env(slippage=0.0001, slippage_prob_adverse=1.0)
    costs = [env._fill_cost(1.0, W) for _ in range(500)]
    assert all(c > 0 for c in costs)
    assert abs(np.mean(costs) - 0.0001) < 0.00003
    env.vol_ratio[:] = 10.0
    hi = np.mean([env._fill_cost(1.0, W) for _ in range(500)])
    assert hi > 2 * np.mean(costs)


def test_swap_charged_once_per_day_rollover():
    env, _, _ = _env(swap_long=-0.0001)
    env.reset()
    swaps = []
    for _ in range(72):
        _, _, _, info = env.step(onehot(1))
        swaps.append(info["swap"])
    charged = [s for s in swaps if s > 0]
    assert len(charged) == 3                                # 72 hourly bars -> 3 rollovers
    assert all(np.isclose(s, 0.0001) or np.isclose(s, 0.0003) for s in charged)
    assert any(np.isclose(s, 0.0003) for s in charged)      # one Wednesday triple swap


def test_stop_loss_forces_flat_and_caps_loss():
    X, r, ts = _data()
    r = r.copy()
    r[W] = -0.05  # 5% crash on the first held bar
    env = RealisticTradingEnv(X, r, timestamps=ts, window=W, spread=0, commission=0,
                              slippage=0, swap_long=0, swap_short=0, stop_loss=0.01,
                              max_drawdown=None, random_start=False, reward_scale=1.0)
    env.reset()
    _, _, done, info = env.step(onehot(1))
    assert info["forced_close"] and info["position"] == 0 and not done
    assert np.isclose(env.equity, 0.99)
    assert env.sl_hits == 1 and env.n_trades == 1


def test_max_drawdown_ends_episode():
    X, r, ts = _data()
    r = r.copy()
    r[W:W + 10] = -0.05
    env = RealisticTradingEnv(X, r, timestamps=ts, window=W, spread=0, commission=0,
                              slippage=0, swap_long=0, swap_short=0, stop_loss=None,
                              max_drawdown=0.2, random_start=False, reward_scale=1.0)
    env.reset()
    done = False
    n = 0
    while not done:
        obs, _, done, info = env.step(onehot(1))
        n += 1
    assert info["blown_up"] and n == 5 and env.pos == 0
    assert not obs.any()


def test_random_start_and_episode_length():
    env, _, _ = _env(random_start=True, max_episode_steps=100, seed=3)
    starts = set()
    for _ in range(5):
        env.reset()
        starts.add(env.t)
        steps = 0
        done = False
        while not done:
            _, _, done, _ = env.step(onehot(0))
            steps += 1
        assert steps == 100
        assert env.t <= T - 1
    assert len(starts) > 1
    assert min(starts) >= W


def test_long_only_mode():
    X, r, ts = _data()
    env = RealisticTradingEnv(X, r, window=W, allow_short=False, random_start=False)
    assert env.action_space == 2
    env.reset()
    _, _, _, info = env.step(onehot(1, 2))
    assert info["position"] == 1
    assert env.day_id is None and env._swap_cost(W + 1, 1) == 0.0


def test_episode_stats_keys():
    env, _, _ = _env(random_start=True, max_episode_steps=50)
    env.reset()
    rng = np.random.default_rng(0)
    done = False
    while not done:
        _, _, done, _ = env.step(onehot(rng.integers(3)))
    st = env.episode_stats()
    for k in ("return_pct", "max_drawdown_pct", "trades", "win_rate", "costs_paid", "reward_sharpe"):
        assert k in st
    assert st["steps"] == 50


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
