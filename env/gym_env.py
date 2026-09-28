from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.dreamer_trading_env import DEFAULT_ENV_KWARGS, RealisticTradingEnv


class GymTradingEnv(gym.Env):
    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(
        self,
        features,
        returns,
        timestamps=None,
        window=64,
        allow_short=True,
        max_episode_steps=4096,
        seed=None,
        **env_kwargs,
    ):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        kwargs = {**DEFAULT_ENV_KWARGS, **env_kwargs}
        kwargs.update(
            window=window,
            allow_short=allow_short,
            max_episode_steps=None,
            seed=seed,
        )
        self.env = RealisticTradingEnv(features, returns, timestamps=timestamps, **kwargs)
        self.action_space = spaces.Discrete(self.env.action_space)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.env.observation_space,),
            dtype=np.float32,
        )
        self._steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._steps = 0
        if seed is not None:
            self.env.rng = np.random.default_rng(seed)
        return self.env.reset(), {}

    def step(self, action):
        action = int(np.asarray(action).item())
        one_hot = np.zeros(self.action_space.n, dtype=np.float32)
        one_hot[action] = 1.0
        obs, reward, terminated, info = self.env.step(one_hot)
        self._steps += 1
        truncated = bool(
            not terminated
            and self.max_episode_steps is not None
            and self._steps >= self.max_episode_steps
        )
        return obs, reward, bool(terminated), truncated, info

    def close(self):
        return None
