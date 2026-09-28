import copy

import numpy as np
import pandas as pd
import torch

from env.dreamer_trading_env import RealisticTradingEnv
from models.dreamer_agent import ReplayBuffer


def build_task_buffer(agent, env_segment, steps, *, seed=None):
    if isinstance(env_segment, RealisticTradingEnv):
        env = env_segment
    else:
        features = np.asarray(env_segment["features"], dtype=np.float32)
        returns = np.asarray(env_segment["returns"], dtype=np.float32)
        window = int(env_segment["window"])
        timestamps = env_segment.get("timestamps")
        env = RealisticTradingEnv(
            features,
            returns,
            timestamps=timestamps,
            window=window,
            allow_short=agent.action_dim == 3,
            max_episode_steps=None,
            random_start=False,
            seed=seed,
        )
    source_buffer = agent.replay_buffer
    agent.replay_buffer = ReplayBuffer(
        capacity=max(source_buffer.capacity, int(steps) + 1),
        seq_len=source_buffer.seq_len,
    )
    agent.prev_action = None
    observation = env.reset()
    h = z = None
    for _ in range(int(steps)):
        action, (h, z) = agent.act(observation, h, z, deterministic=False)
        next_observation, reward, done, _ = env.step(action)
        agent.replay_buffer.add(observation, action, reward, done)
        if done:
            observation = env.reset()
            h = z = None
            agent.prev_action = None
        else:
            observation = next_observation
    return agent.replay_buffer


class MarketRegimeGenerator:
    @staticmethod
    def _labels(returns):
        returns = pd.Series(np.asarray(returns, dtype=np.float64))
        past = returns.shift(1)
        trend_return = past.rolling(120, min_periods=120).sum().to_numpy()
        trend_volatility = past.rolling(120, min_periods=120).std(ddof=0).to_numpy()
        volatility_48 = past.rolling(48, min_periods=48).std(ddof=0).to_numpy()
        expanding_median = (
            pd.Series(volatility_48)
            .expanding(min_periods=1)
            .median()
            .shift(1)
            .to_numpy()
        )
        labels = np.full(len(returns), "unknown", dtype=object)
        for index in range(len(returns)):
            volatility = volatility_48[index]
            median = expanding_median[index]
            trend = trend_return[index]
            long_vol = trend_volatility[index]
            if not np.isfinite([volatility, median, trend, long_vol]).all():
                continue
            if median > 0 and volatility > 1.5 * median:
                labels[index] = "high_vol"
            elif median > 0 and volatility < 0.5 * median:
                labels[index] = "low_vol"
            elif long_vol > 0 and abs(trend) > 0.5 * long_vol * np.sqrt(120):
                labels[index] = "trend_up" if trend > 0 else "trend_down"
            else:
                labels[index] = "range"
        return labels

    @staticmethod
    def _find_label_periods(labels, accepted, min_len=1):
        labels = np.asarray(labels)
        accepted = {accepted} if isinstance(accepted, str) else set(accepted)
        periods = []
        start = 0
        while start < len(labels):
            label = labels[start]
            end = start + 1
            while end < len(labels) and labels[end] == label:
                end += 1
            if label in accepted and end - start >= min_len:
                periods.append((start, end))
            start = end
        return periods

    @staticmethod
    def _find_trending_periods(labels, min_len=1):
        return MarketRegimeGenerator._find_label_periods(
            labels, {"trend_up", "trend_down"}, min_len
        )

    @staticmethod
    def _find_ranging_periods(labels, min_len=1):
        return MarketRegimeGenerator._find_label_periods(
            labels, "range", min_len
        )

    @staticmethod
    def _find_volatile_periods(labels, min_len=1):
        return MarketRegimeGenerator._find_label_periods(
            labels, {"high_vol", "low_vol"}, min_len
        )

    @classmethod
    def generate_regimes(cls, features, returns, timestamps, window, min_len=512):
        features = np.asarray(features, dtype=np.float32)
        returns = np.asarray(returns, dtype=np.float32)
        timestamps = np.asarray(timestamps)
        if features.ndim != 2 or returns.ndim != 1:
            raise ValueError("features must be 2D and returns must be 1D")
        if len(features) != len(returns) or len(features) != len(timestamps):
            raise ValueError("features, returns, and timestamps must have equal lengths")
        labels = cls._labels(returns)
        tasks = []
        for label in ("trend_up", "trend_down", "range", "high_vol", "low_vol"):
            for start, end in cls._find_label_periods(labels, label, min_len):
                split = start + int((end - start) * 0.7)
                if split <= start or split >= end:
                    continue
                tasks.append(
                    {
                        "name": label,
                        "label": label,
                        "start": start,
                        "end": end,
                        "support": {
                            "features": features[start:split],
                            "returns": returns[start:split],
                            "timestamps": timestamps[start:split],
                            "window": int(window),
                        },
                        "query": {
                            "features": features[split:end],
                            "returns": returns[split:end],
                            "timestamps": timestamps[split:end],
                            "window": int(window),
                        },
                    }
                )
        return tasks


class MAMLTrader:
    def __init__(
        self,
        agent,
        meta_lr=1e-4,
        adapt_lr=1e-3,
        adapt_steps=3,
        first_order=True,
    ):
        if not first_order:
            raise NotImplementedError("only first-order MAML is supported")
        self.base_agent = agent
        self.agent = self.base_agent
        self.meta_lr = float(meta_lr)
        self.adapt_lr = float(adapt_lr)
        self.adapt_steps = int(adapt_steps)
        self.first_order = True
        self.meta_optimizer = torch.optim.Adam(
            self.base_agent.world_model_params, lr=self.meta_lr
        )
        self.sequence_length = int(self.base_agent.replay_buffer.seq_len)

    def _buffer(self, data):
        if hasattr(data, "sample"):
            return data
        if {"obs", "action", "reward", "done"}.issubset(data):
            return data
        raise ValueError("task data must be a replay buffer or a market segment")

    def _sample_batch(self, buffer, batch_size=32):
        if hasattr(buffer, "sample"):
            batch = buffer.sample(batch_size)
            if batch is None:
                raise ValueError("replay buffer does not contain a sampleable sequence")
            return batch
        sequence_length = self.sequence_length
        size = len(buffer["obs"])
        max_start = size - sequence_length
        if max_start < 0:
            raise ValueError(
                f"buffer has {size} transitions, requires {sequence_length}"
            )
        done = np.asarray(buffer["done"], dtype=np.float32)
        valid = [
            start
            for start in range(max_start + 1)
            if not np.any(done[start:start + sequence_length - 1])
        ]
        if not valid:
            raise ValueError("buffer has no sequence that stays within an episode")
        starts = np.random.choice(
            valid,
            size=int(batch_size),
            replace=len(valid) < int(batch_size),
        )
        indices = starts[:, None] + np.arange(sequence_length)[None, :]
        return {
            key: np.asarray(value)[indices]
            for key, value in buffer.items()
            if key in {"obs", "action", "reward", "done"}
        }

    def _adapt_copy(self, agent, support_buffer, steps, batch_size):
        params = agent.world_model_params
        losses = []
        for _ in range(int(steps)):
            batch = self._sample_batch(support_buffer, batch_size)
            loss, _ = agent.compute_world_model_loss(batch)
            gradients = torch.autograd.grad(loss, params)
            with torch.no_grad():
                for parameter, gradient in zip(params, gradients):
                    parameter.add_(gradient, alpha=-self.adapt_lr)
            losses.append(float(loss.detach()))
        return losses

    def meta_train(
        self,
        market_regimes,
        num_epochs=1,
        tasks_per_batch=4,
        batch_size=32,
    ):
        if not market_regimes:
            raise ValueError("meta-training requires at least one task")
        losses = []
        for _ in range(int(num_epochs)):
            count = min(int(tasks_per_batch), len(market_regimes))
            task_indices = np.random.choice(len(market_regimes), count, replace=False)
            self.meta_optimizer.zero_grad(set_to_none=True)
            task_losses = []
            support_losses = []
            for task_index in task_indices:
                task = market_regimes[int(task_index)]
                adapted = copy.deepcopy(self.base_agent)
                support_segment = task["support"]
                support_steps = len(support_segment["returns"]) - int(support_segment["window"]) - 1
                support_buffer = build_task_buffer(
                    adapted,
                    support_segment,
                    support_steps,
                )
                support_loss = self._adapt_copy(
                    adapted, support_buffer, self.adapt_steps, batch_size
                )
                query_agent = copy.deepcopy(adapted)
                query_segment = task["query"]
                query_steps = len(query_segment["returns"]) - int(query_segment["window"]) - 1
                query_buffer = build_task_buffer(
                    query_agent,
                    query_segment,
                    query_steps,
                )
                query_batch = self._sample_batch(query_buffer, batch_size)
                query_loss, _ = query_agent.compute_world_model_loss(query_batch)
                query_gradients = torch.autograd.grad(
                    query_loss, query_agent.world_model_params
                )
                for parameter, gradient in zip(
                    self.base_agent.world_model_params, query_gradients
                ):
                    if parameter.grad is None:
                        parameter.grad = gradient.detach().clone() / count
                    else:
                        parameter.grad.add_(gradient.detach(), alpha=1.0 / count)
                task_losses.append(float(query_loss.detach()))
                support_losses.extend(support_loss)
            torch.nn.utils.clip_grad_norm_(
                self.base_agent.world_model_params, max_norm=1000.0
            )
            self.meta_optimizer.step()
            losses.append(
                {
                    "meta_loss": float(np.mean(task_losses)),
                    "support_loss": float(np.mean(support_losses)) if support_losses else 0.0,
                }
            )
        return losses

    def fast_adapt(self, buffer, steps=None, batch_size=32):
        adapted = copy.deepcopy(self.base_agent)
        if not hasattr(buffer, "sample") and not {"obs", "action", "reward", "done"}.issubset(buffer):
            steps_for_buffer = int(steps or self.adapt_steps)
            buffer = build_task_buffer(adapted, buffer, steps_for_buffer)
        support_buffer = self._buffer(buffer)
        self._adapt_copy(
            adapted,
            support_buffer,
            self.adapt_steps if steps is None else int(steps),
            batch_size,
        )
        return adapted
