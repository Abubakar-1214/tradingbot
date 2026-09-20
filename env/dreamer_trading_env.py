"""
Realistic XAUUSD trading environment for DreamerV3 training.

Replaces the toy "flat/long, fixed 1bp cost" environment used by the Dreamer
training scripts with a broker-like simulation:

- Actions: flat / long / short (short can be disabled).
- Execution: bid/ask spread, commission, random volatility-scaled slippage,
  spread widening in high-volatility bars and (optionally) around news events.
- Financing: overnight swap charged once per calendar day rollover (triple on
  Wednesday, like most FX/CFD brokers) when timestamps are available.
- Risk: per-trade stop-loss / take-profit (checked against the bar's move),
  leverage, and a maximum-drawdown circuit breaker that ends the episode.
- Episodes: random start offset and bounded length so the replay buffer sees
  many market regimes instead of one multi-year episode.
- Observation: feature window plus an account-state vector
  [position, unrealized trade pnl, bars in trade, drawdown, log equity].
- Reward: leveraged log-return of equity (scaled), optionally penalised by
  drawdown, so the agent optimises risk-adjusted compounding, not raw ticks.

Causality: at step t the agent sees features up to bar t-1; the position
chosen at t is filled at the open of bar t and earns returns[t]. Nothing from
bar t or later is in the observation.

Interface matches the Dreamer scripts: ``reset() -> obs`` and
``step(action_onehot) -> (obs, reward, done, info)``.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

ACCOUNT_STATE_DIM = 5

# Shared defaults for the Dreamer training/evaluation scripts. Costs are
# fractions of notional: 1e-4 = 1 bp = $0.20 on $2000 gold.
DEFAULT_ENV_KWARGS = dict(
    window=64,
    allow_short=True,
    leverage=1.0,
    spread=0.00025,        # 2.5 bp round-trip spread (~$0.50)
    commission=0.00003,    # 0.3 bp per side
    slippage=0.00005,      # mean |slippage| per fill, vol-scaled, mostly adverse
    swap_long=-0.00004,    # daily financing, fraction of notional
    swap_short=-0.00002,
    stop_loss=0.01,        # 1% adverse move on a trade -> forced close
    take_profit=None,
    max_drawdown=0.30,     # 30% drawdown from peak ends the episode
    drawdown_penalty=0.0,
    reward_scale=100.0,    # 1% equity change -> reward 1.0
    max_episode_steps=4096,
    random_start=True,
)

# Evaluation: one continuous pass over the period, same costs, no circuit breaker.
EVAL_ENV_OVERRIDES = dict(max_episode_steps=None, random_start=False, max_drawdown=None)


class RealisticTradingEnv:
    """Broker-like XAUUSD environment with a Dreamer-compatible interface."""

    def __init__(
        self,
        features,
        returns,
        timestamps=None,
        window=64,
        allow_short=True,
        leverage=1.0,
        spread=0.00025,          # 2.5 bps ~ $0.50 on $2000 gold
        commission=0.00003,      # 0.3 bps per side
        slippage=0.00005,        # mean |slippage| per fill, vol-scaled
        slippage_prob_adverse=0.8,
        spread_vol_multiplier=2.5,
        vol_window=100,
        event_mask=None,
        event_spread_multiplier=2.0,
        event_slippage_multiplier=3.0,
        swap_long=-0.00004,      # daily financing as fraction of notional
        swap_short=-0.00002,
        stop_loss=0.01,          # 1% adverse move on the trade -> forced close
        take_profit=None,        # e.g. 0.02; None disables
        max_drawdown=0.30,       # episode ends when equity falls 30% from peak
        drawdown_penalty=0.0,    # reward -= penalty * drawdown increase
        reward_scale=100.0,      # 1% equity change -> reward 1.0
        max_episode_steps=4096,
        random_start=True,
        seed=None,
    ):
        assert features.ndim == 2 and returns.ndim == 1
        assert len(features) == len(returns)

        self.X = np.ascontiguousarray(features, dtype=np.float32)
        self.r = np.ascontiguousarray(returns, dtype=np.float32)
        self.T = len(self.r)
        self.window = int(window)
        self.allow_short = bool(allow_short)
        self.leverage = float(leverage)

        self.spread = float(spread)
        self.commission = float(commission)
        self.slippage = float(slippage)
        self.slippage_prob_adverse = float(slippage_prob_adverse)
        self.spread_vol_multiplier = float(spread_vol_multiplier)
        self.event_spread_multiplier = float(event_spread_multiplier)
        self.event_slippage_multiplier = float(event_slippage_multiplier)
        self.swap_long = float(swap_long)
        self.swap_short = float(swap_short)
        self.stop_loss = None if stop_loss is None else float(stop_loss)
        self.take_profit = None if take_profit is None else float(take_profit)
        self.max_drawdown = None if max_drawdown is None else float(max_drawdown)
        self.drawdown_penalty = float(drawdown_penalty)
        self.reward_scale = float(reward_scale)
        self.max_episode_steps = None if max_episode_steps is None else int(max_episode_steps)
        self.random_start = bool(random_start)
        self.rng = np.random.default_rng(seed)

        if self.T <= self.window + 2:
            raise ValueError(f"Need more than window+2={self.window + 2} bars, got {self.T}")

        self.vol_ratio = self._rolling_vol_ratio(self.r, int(vol_window))

        if event_mask is None:
            self.event_mask = np.zeros(self.T, dtype=bool)
        else:
            self.event_mask = np.asarray(event_mask, dtype=bool)
            assert len(self.event_mask) == self.T

        self.day_id = self._day_ids(timestamps)
        self.weekday = self._weekdays(timestamps)

        self.action_space = 3 if self.allow_short else 2
        self.observation_space = self.window * self.X.shape[1] + ACCOUNT_STATE_DIM

        logger.info("RealisticTradingEnv: T=%d features=%d window=%d actions=%d "
                    "spread=%.1fbp comm=%.1fbp slip=%.1fbp lev=%.1f SL=%s TP=%s maxDD=%s",
                    self.T, self.X.shape[1], self.window, self.action_space,
                    self.spread * 1e4, self.commission * 1e4, self.slippage * 1e4,
                    self.leverage, self.stop_loss, self.take_profit, self.max_drawdown)

        self.reset()

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _rolling_vol_ratio(r, vol_window):
        """Trailing std of returns (up to and excluding bar t) / expanding mean of that std.

        Both numerator and denominator only use bars before t, so a fill's cost
        never changes when later data is appended.
        """
        r64 = r.astype(np.float64)
        csum = np.cumsum(np.insert(r64, 0, 0.0))
        csum2 = np.cumsum(np.insert(r64 ** 2, 0, 0.0))
        n = np.arange(len(r64))
        lo = np.maximum(0, n - vol_window)
        cnt = np.maximum(n - lo, 1)
        mean = (csum[n] - csum[lo]) / cnt
        var = (csum2[n] - csum2[lo]) / cnt - mean ** 2
        vol = np.sqrt(np.maximum(var, 0.0))
        baseline = np.cumsum(vol) / np.maximum(n, 1)
        ratio = np.divide(vol, baseline, out=np.ones_like(vol), where=baseline > 0)
        ratio[:vol_window] = 1.0
        return ratio.astype(np.float32)

    def _day_ids(self, timestamps):
        if timestamps is None:
            return None
        ts = np.asarray(timestamps).astype("datetime64[D]")
        assert len(ts) == self.T
        return ts.astype(np.int64)

    def _weekdays(self, timestamps):
        if timestamps is None:
            return None
        days = np.asarray(timestamps).astype("datetime64[D]").astype(np.int64)
        # 1970-01-01 was a Thursday (Monday=0 -> 3)
        return ((days + 3) % 7).astype(np.int64)

    # --------------------------------------------------------------- episode
    def reset(self):
        if self.random_start and self.max_episode_steps is not None:
            hi = self.T - 1 - min(self.max_episode_steps, self.T - 1 - self.window)
            self.t = int(self.rng.integers(self.window, max(self.window, hi) + 1))
        else:
            self.t = self.window

        self.pos = 0
        self.equity = 1.0
        self.peak_equity = 1.0
        self.drawdown = 0.0
        self.max_dd = 0.0
        self.steps = 0

        self.entry_equity = 1.0
        self.trade_pnl = 0.0
        self.bars_in_trade = 0

        self.n_trades = 0
        self.n_wins = 0
        self.total_costs = 0.0
        self.total_swap = 0.0
        self.sl_hits = 0
        self.tp_hits = 0
        self.rewards = []
        return self._get_obs()

    def _account_state(self):
        return np.array([
            float(self.pos),
            float(np.clip(self.trade_pnl * 100.0, -10.0, 10.0)),
            float(np.log1p(self.bars_in_trade)) / 5.0,
            float(self.drawdown),
            float(np.log(max(self.equity, 1e-6))),
        ], dtype=np.float32)

    def _get_obs(self):
        w = self.X[self.t - self.window:self.t]
        return np.concatenate([w.reshape(-1), self._account_state()]).astype(np.float32)

    # ------------------------------------------------------------- execution
    def _decode_action(self, action_onehot):
        idx = int(np.argmax(action_onehot))
        if self.allow_short:
            return {0: 0, 1: 1, 2: -1}[idx]
        return 1 if idx == 1 else 0

    def _fill_cost(self, size_change, t):
        """Fractional cost (of notional traded) for changing exposure by |size_change|."""
        if size_change == 0:
            return 0.0
        vol_ratio = float(self.vol_ratio[t])
        spread = self.spread
        slip = self.slippage
        if vol_ratio > 1.5:
            m = min(vol_ratio, self.spread_vol_multiplier)
            spread *= m
            slip *= m
        if self.event_mask[t]:
            spread *= self.event_spread_multiplier
            slip *= self.event_slippage_multiplier

        slip_draw = abs(self.rng.normal(slip, 0.5 * slip))
        if self.rng.random() < self.slippage_prob_adverse:
            slip_signed = slip_draw
        else:
            slip_signed = -0.5 * slip_draw  # price improvement

        per_unit = 0.5 * spread + self.commission + slip_signed
        return per_unit * abs(size_change)

    def _swap_cost(self, t, pos):
        if pos == 0 or self.day_id is None or t == 0:
            return 0.0
        if self.day_id[t] == self.day_id[t - 1]:
            return 0.0
        nights = 3 if self.weekday is not None and self.weekday[t - 1] == 2 else 1
        rate = self.swap_long if pos > 0 else self.swap_short
        return -rate * nights  # positive number = cost

    def _debit(self, frac):
        """Charge a fractional fill cost against current equity; returns the cash amount."""
        if frac == 0.0:
            return 0.0
        cash = frac * self.equity
        self.equity = max(self.equity * (1.0 - frac), 1e-6)
        self.total_costs += cash
        return cash

    def _liquidate(self, pos, t):
        """Close an open position at bar t: pay the exit fill, then book the trade."""
        cost = self._fill_cost(pos * self.leverage, t)
        self._debit(cost)
        self._close_trade()
        return cost

    def step(self, action_onehot):
        prev_pos = self.pos
        new_pos = self._decode_action(action_onehot)
        t = self.t
        ret = float(self.r[t])
        equity_before = self.equity
        cost = 0.0

        # 1. Rollover swap on exposure held from the previous bar into this one.
        swap = self._swap_cost(t, prev_pos)
        if swap:
            self.total_swap += swap * self.equity
            self.equity = max(self.equity * (1.0 - swap), 1e-6)

        # 2. Execute position change at bar open: exit the old side, then enter the new.
        if new_pos != prev_pos:
            if prev_pos != 0:
                cost += self._liquidate(prev_pos, t)
            if new_pos != 0:
                cost += self._debit(self._fill_cost(new_pos * self.leverage, t))
                self._open_trade()

        # 3. Hold through the bar; SL/TP are checked against the bar's move.
        exposure = new_pos * self.leverage
        gross = exposure * ret
        forced_close = False
        if new_pos != 0:
            trade_move = self.trade_pnl + gross  # cumulative leveraged pnl of the trade
            if self.stop_loss is not None and trade_move <= -self.stop_loss:
                gross = -self.stop_loss - self.trade_pnl
                forced_close = True
                self.sl_hits += 1
            elif self.take_profit is not None and trade_move >= self.take_profit:
                gross = self.take_profit - self.trade_pnl
                forced_close = True
                self.tp_hits += 1

        self.equity = max(self.equity * (1.0 + gross), 1e-6)
        if new_pos != 0:
            self.trade_pnl += gross
            self.bars_in_trade += 1

        if forced_close:
            cost += self._liquidate(new_pos, t)
            new_pos = 0

        # 4. Termination; any position still open at episode end is liquidated.
        self.t += 1
        self.steps += 1
        out_of_data = self.t >= self.T - 1
        truncated = self.max_episode_steps is not None and self.steps >= self.max_episode_steps
        blown_up = (self.max_drawdown is not None
                    and 1.0 - self.equity / max(self.peak_equity, self.equity) >= self.max_drawdown)
        done = bool(blown_up or out_of_data or truncated)
        if done and new_pos != 0:
            cost += self._liquidate(new_pos, t)
            new_pos = 0
        self.pos = new_pos

        # 5. Drawdown bookkeeping and reward.
        self.peak_equity = max(self.peak_equity, self.equity)
        prev_dd = self.drawdown
        self.drawdown = 1.0 - self.equity / self.peak_equity
        self.max_dd = max(self.max_dd, self.drawdown)
        log_ret = np.log(self.equity / equity_before)
        reward = log_ret - self.drawdown_penalty * max(self.drawdown - prev_dd, 0.0)
        reward = float(reward * self.reward_scale)
        self.rewards.append(reward)

        obs = self._get_obs() if not done else np.zeros(self.observation_space, dtype=np.float32)

        info = {
            "equity": self.equity,
            "position": self.pos,
            "pnl": self.equity / equity_before - 1.0,
            "return": ret,
            "cost": cost / equity_before,  # fraction of pre-step equity
            "swap": swap,
            "drawdown": self.drawdown,
            "forced_close": forced_close,
            "blown_up": blown_up,
        }
        return obs, reward, done, info

    def _open_trade(self):
        self.entry_equity = self.equity
        self.trade_pnl = 0.0
        self.bars_in_trade = 0

    def _close_trade(self):
        self.n_trades += 1
        if self.equity > self.entry_equity:
            self.n_wins += 1
        self.trade_pnl = 0.0
        self.bars_in_trade = 0

    # ------------------------------------------------------------------ stats
    def episode_stats(self):
        rewards = np.asarray(self.rewards, dtype=np.float64)
        sharpe = 0.0
        if len(rewards) > 1 and rewards.std() > 0:
            sharpe = float(rewards.mean() / rewards.std() * np.sqrt(len(rewards)))
        return {
            "steps": self.steps,
            "equity": float(self.equity),
            "return_pct": float((self.equity - 1.0) * 100.0),
            "max_drawdown_pct": float(self.max_dd * 100.0),
            "trades": self.n_trades,
            "win_rate": float(self.n_wins / self.n_trades) if self.n_trades else 0.0,
            "sl_hits": self.sl_hits,
            "tp_hits": self.tp_hits,
            "costs_paid": float(self.total_costs),
            "swap_paid": float(self.total_swap),
            "reward_sharpe": sharpe,
        }
