from collections import deque

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class MarketMakerAgent:
    def __init__(
        self,
        state_dim,
        action_dim=4,
        hidden_dim=128,
        learning_rate=3e-4,
        gamma=0.99,
        baseline_momentum=0.9,
        manip_cost=0.01,
        max_manipulation_rate=0.2,
    ):
        if action_dim != 4:
            raise ValueError("MarketMakerAgent requires exactly four actions")
        if not 0.0 <= max_manipulation_rate <= 1.0:
            raise ValueError("max_manipulation_rate must be between 0 and 1")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.baseline_momentum = float(baseline_momentum)
        self.manip_cost = float(manip_cost)
        self.max_manipulation_rate = float(max_manipulation_rate)
        self.policy = nn.Sequential(
            nn.Linear(self.state_dim + 10, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_dim),
        )
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.trader_history = deque(maxlen=100)
        self.baseline = 0.0
        self.total_profit = 0.0
        self.successful_traps = 0
        self.begin_episode()

    def begin_episode(self):
        self.total_decisions = 0
        self.manipulation_count = 0
        self._episode_log_probs = []
        self._episode_rewards = []

    @property
    def manipulation_rate(self):
        return self.manipulation_count / max(self.total_decisions, 1)

    @staticmethod
    def _action_index(action):
        if hasattr(action, "action"):
            return int(action.action)
        values = np.asarray(action)
        return int(values.item()) if values.ndim == 0 else int(np.argmax(values))

    def _pattern_features(self):
        recent = np.asarray(list(self.trader_history)[-20:], dtype=np.float32)
        if len(recent) == 0:
            return np.zeros(10, dtype=np.float32)
        changes = np.count_nonzero(recent[1:] != recent[:-1]) if len(recent) > 1 else 0
        direction = recent - 1.0
        repeats = {}
        for index in range(max(0, len(recent) - 3)):
            pattern = tuple(recent[index:index + 3])
            repeats[pattern] = repeats.get(pattern, 0) + 1
        predictability = (
            max(repeats.values()) / max(len(recent) - 2, 1) if repeats else 0.0
        )
        return np.asarray(
            [
                float(np.mean(recent == 1)),
                float(np.mean(recent == 2)),
                float(np.mean(direction)),
                float(changes / max(len(recent) - 1, 1)),
                float(predictability),
                float(np.mean(direction[-5:]) - np.mean(direction[:5])),
                float(np.std(direction)),
                float(np.mean(recent == 0)),
                float(direction[-1]),
                float(len(set(recent[-5:])) == 1),
            ],
            dtype=np.float32,
        )

    def select_action(self, trader_action, market_state):
        trader_action = self._action_index(trader_action)
        self.trader_history.append(trader_action)
        market_state = np.asarray(market_state, dtype=np.float32).reshape(-1)
        if market_state.size != self.state_dim:
            raise ValueError(
                f"expected market state width {self.state_dim}, got {market_state.size}"
            )
        observation = np.concatenate((market_state, self._pattern_features()))
        state_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        probabilities = torch.softmax(self.policy(state_tensor), dim=-1).squeeze(0)
        next_decision = self.total_decisions + 1
        allowed_count = int(
            np.floor(self.max_manipulation_rate * next_decision + 1e-12)
        )
        if self.manipulation_count >= allowed_count:
            probabilities = torch.cat(
                (probabilities[:1], torch.zeros_like(probabilities[1:]))
            )
            probabilities = probabilities / probabilities.sum()
        distribution = Categorical(probs=probabilities)
        action = distribution.sample()
        log_probability = distribution.log_prob(action)
        action_index = int(action.item())
        self.total_decisions += 1
        if action_index:
            self.manipulation_count += 1
        return action_index, log_probability

    def respond(self, trader_action, market_state, trader_pattern=None):
        action, log_probability = self.select_action(trader_action, market_state)
        self._pending_log_probability = log_probability
        return action

    def observe_reward(self, log_probability, reward):
        self._episode_log_probs.append(log_probability)
        self._episode_rewards.append(float(reward))
        self.total_profit += float(reward)
        if reward > 0:
            self.successful_traps += 1

    def finish_episode(self):
        if not self._episode_rewards:
            return 0.0
        returns = []
        running = 0.0
        for reward in reversed(self._episode_rewards):
            running = reward + self.gamma * running
            returns.append(running)
        returns = torch.as_tensor(list(reversed(returns)), dtype=torch.float32)
        baseline = self.baseline
        advantages = returns - baseline
        loss = -(
            torch.stack(self._episode_log_probs) * advantages.detach()
        ).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        self.baseline = (
            self.baseline_momentum * baseline
            + (1.0 - self.baseline_momentum) * float(returns.mean())
        )
        mean_reward = float(np.mean(self._episode_rewards))
        self._episode_log_probs = []
        self._episode_rewards = []
        return mean_reward

    def learn(self, reward):
        self.total_profit += float(reward)
        if reward > 0:
            self.successful_traps += 1

    def get_statistics(self):
        return {
            "total_profit": self.total_profit,
            "successful_traps": self.successful_traps,
            "avg_profit_per_trap": self.total_profit / max(1, self.successful_traps),
            "manipulation_rate": self.manipulation_rate,
        }


class AdversarialTradingEnv:
    def __init__(self, base_env, market_maker):
        self.base_env = base_env
        self.market_maker = market_maker
        self.current_manipulation = 0
        self.action_names = ("none", "widen_spread", "slippage", "stop_hunt")

    def _get_market_state(self):
        return self.base_env._get_obs().copy()

    @staticmethod
    def _action_index(action):
        if hasattr(action, "action"):
            return int(action.action)
        values = np.asarray(action)
        return int(values.item()) if values.ndim == 0 else int(np.argmax(values))

    def _action_one_hot(self, action):
        action = np.asarray(action)
        if action.ndim == 0:
            one_hot = np.zeros(self.base_env.action_space, dtype=np.float32)
            one_hot[int(action)] = 1.0
            return one_hot
        if action.size != self.base_env.action_space:
            raise ValueError("trader action vector does not match environment action space")
        return action.astype(np.float32, copy=False).reshape(-1)

    def step(self, trader_action):
        state = self._get_market_state()
        trader_action_index = self._action_index(trader_action)
        mm_action, log_probability = self.market_maker.select_action(
            trader_action_index, state
        )
        stop_loss = self.base_env.stop_loss or 0.0
        perturbation = {
            "spread_mult": 3.0 if mm_action == 1 else 1.0,
            "slippage_mult": 3.0 if mm_action == 2 else 1.0,
            "adverse_gap": 0.5 * stop_loss if mm_action == 3 else 0.0,
        }
        self.base_env.set_perturbation(**perturbation)
        obs, trader_reward, done, info = self.base_env.step(
            self._action_one_hot(trader_action)
        )
        budget_cost = self.market_maker.manip_cost if mm_action else 0.0
        mm_reward = -float(trader_reward) - budget_cost
        self.market_maker.observe_reward(log_probability, mm_reward)
        self.current_manipulation = mm_action
        info.update(
            mm_action=mm_action,
            mm_profit=mm_reward,
            budget_cost=budget_cost,
            manipulation_type=self.action_names[mm_action],
        )
        return obs, trader_reward, done, info

    def reset(self):
        self.current_manipulation = 0
        self.market_maker.begin_episode()
        return self.base_env.reset()


class SelfPlayTrainer:
    def __init__(
        self,
        trader_agent,
        mm_agent,
        env,
        train_every=4,
        batch_size=16,
    ):
        self.trader = trader_agent
        self.mm = mm_agent
        self.env = env
        self.adv_env = AdversarialTradingEnv(env, mm_agent)
        self.train_every = max(1, int(train_every))
        self.batch_size = int(batch_size)
        self.history = {
            "trader_wins": [],
            "mm_profits": [],
            "epochs": 0,
            "trader_updates": 0,
        }
        self._h = None
        self._z = None
        self._steps_since_update = 0

    def _reset_trader_state(self):
        self._h = None
        self._z = None
        if hasattr(self.trader, "prev_action"):
            self.trader.prev_action = None

    def _trader_action(self, obs, deterministic=False):
        result = self.trader.act(
            obs,
            self._h,
            self._z,
            deterministic=deterministic,
        )
        action, state = result
        if isinstance(state, tuple) and len(state) == 2:
            self._h, self._z = state
        return np.asarray(action, dtype=np.float32).reshape(-1)

    def _train_trader(self, steps):
        obs = self.adv_env.reset()
        self._reset_trader_state()
        total_reward = 0.0
        episode_open = True
        for step in range(int(steps)):
            action = self._trader_action(obs)
            next_obs, reward, done, _ = self.adv_env.step(action)
            self.trader.replay_buffer.add(obs, action, reward, done)
            total_reward += reward
            self._steps_since_update += 1
            if self._steps_since_update >= self.train_every:
                self._steps_since_update = 0
                if len(self.trader.replay_buffer) >= self.trader.replay_buffer.seq_len + 1:
                    metrics = self.trader.train_step(batch_size=self.batch_size)
                    if metrics is not None:
                        self.history["trader_updates"] += 1
            obs = next_obs
            if done:
                self.mm.finish_episode()
                episode_open = False
                self._reset_trader_state()
                if step + 1 < steps:
                    obs = self.adv_env.reset()
                    episode_open = True
        if episode_open:
            self.mm.finish_episode()
        return total_reward / max(int(steps), 1)

    def _train_mm(self, steps):
        obs = self.adv_env.reset()
        self._reset_trader_state()
        total_profit = 0.0
        episode_open = True
        for step in range(int(steps)):
            action = self._trader_action(obs, deterministic=True)
            obs, _, done, info = self.adv_env.step(action)
            total_profit += info["mm_profit"]
            if done:
                self.mm.finish_episode()
                episode_open = False
                self._reset_trader_state()
                if step + 1 < steps:
                    obs = self.adv_env.reset()
                    episode_open = True
        if episode_open:
            self.mm.finish_episode()
        return total_profit / max(int(steps), 1)

    def train(self, num_epochs=100, steps_per_epoch=1000):
        for _ in range(int(num_epochs)):
            trader_reward = self._train_trader(steps_per_epoch)
            mm_profit = self._train_mm(steps_per_epoch)
            self.history["trader_wins"].append(trader_reward)
            self.history["mm_profits"].append(mm_profit)
            self.history["epochs"] += 1
        return self.history
