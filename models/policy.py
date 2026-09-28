from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from stable_baselines3 import PPO

from models.dreamer_agent import DreamerV3Agent
from models.dreamer_components import symexp


@dataclass(frozen=True)
class PolicyOutput:
    action: int
    probs: np.ndarray
    confidence: float
    info: dict = field(default_factory=dict)


@runtime_checkable
class TradingPolicy(Protocol):
    action_dim: int
    obs_dim: int

    def reset(self) -> None: ...

    def act(self, obs: np.ndarray) -> PolicyOutput: ...

    def observe_executed(self, action: int) -> None: ...


class PpoPolicy:
    def __init__(self, model_path):
        self.model = PPO.load(str(model_path), device="cpu")
        self.action_dim = int(self.model.action_space.n)
        self.obs_dim = int(self.model.observation_space.shape[0])

    def reset(self) -> None:
        return None

    def act(self, obs: np.ndarray) -> PolicyOutput:
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).reshape(1, -1)
        with torch.no_grad():
            dist = self.model.policy.get_distribution(obs_t).distribution
            probs = dist.probs.detach().cpu().numpy().reshape(-1).astype(np.float64)
        probs /= probs.sum()
        action = int(np.argmax(probs))
        return PolicyOutput(action, probs, float(probs[action]), {})

    def observe_executed(self, action: int) -> None:
        return None


class DreamerPolicy:
    def __init__(self, agent: DreamerV3Agent):
        self.agent = agent
        self.action_dim = int(agent.action_dim)
        self.obs_dim = int(agent.obs_dim)
        self.reset()

    def reset(self) -> None:
        self.h = None
        self.z = None
        self.prev_action = None
        for network in (
            self.agent.encoder,
            self.agent.rssm,
            self.agent.actor,
            self.agent.critic,
        ):
            network.eval()

    def act(self, obs: np.ndarray) -> PolicyOutput:
        obs_t = torch.as_tensor(
            np.asarray(obs, dtype=np.float32), device=self.agent.device
        ).reshape(1, -1)
        if obs_t.shape[-1] != self.obs_dim:
            raise ValueError(f"expected observation width {self.obs_dim}, got {obs_t.shape[-1]}")
        with torch.no_grad():
            embed = self.agent.encoder(obs_t)
            if self.h is None or self.z is None:
                self.h, self.z = self.agent.rssm.initial_state(1, self.agent.device)
                prev_action = torch.zeros(1, self.action_dim, device=self.agent.device)
            else:
                prev_action = self.prev_action
            self.h, self.z, _, _ = self.agent.rssm.observe(
                embed, prev_action, self.h, self.z
            )
            state = self.agent.rssm.get_state(self.h, self.z)
            probs = self.agent.policy_probs(state).detach().cpu().numpy()[0]
            probs = probs.astype(np.float64)
            probs /= probs.sum()
            value = float(symexp(self.agent.critic(state).float()).item())
            latent = torch.cat((self.h, self.z), dim=-1).cpu().numpy()[0].copy()
        action = int(np.argmax(probs))
        return PolicyOutput(
            action,
            probs,
            float(probs[action]),
            {"value": value, "latent": latent},
        )

    def observe_executed(self, action: int) -> None:
        if not 0 <= int(action) < self.action_dim:
            raise ValueError(f"action {action} outside [0, {self.action_dim})")
        self.prev_action = torch.nn.functional.one_hot(
            torch.tensor([int(action)], device=self.agent.device),
            num_classes=self.action_dim,
        ).float()


class TransformerPolicy:
    def __init__(self, agent):
        self.agent = agent
        self.action_dim = int(agent.action_dim)
        self.obs_dim = int(agent.obs_dim)

    def reset(self) -> None:
        return None

    def act(self, obs: np.ndarray) -> PolicyOutput:
        action, logp, value = self.agent.act(obs, deterministic=True)
        probs = self.agent.policy_probs(
            np.asarray(obs, dtype=np.float32).reshape(1, -1)
        )[0].detach().cpu().numpy().astype(np.float64)
        probs /= probs.sum()
        return PolicyOutput(
            int(action),
            probs,
            float(probs[action]),
            {"logp": float(logp), "value": float(value)},
        )

    def observe_executed(self, action: int) -> None:
        if not 0 <= int(action) < self.action_dim:
            raise ValueError(f"action {action} outside [0, {self.action_dim})")


class DreamerMCTSPolicy:
    def __init__(self, dreamer_agent, num_simulations=32, c_puct=1.0):
        from models.mcts import DreamerMCTSAgent

        self.agent = DreamerMCTSAgent(
            dreamer_agent,
            num_simulations=num_simulations,
            c_puct=c_puct,
        )
        self.action_dim = int(dreamer_agent.action_dim)
        self.obs_dim = int(dreamer_agent.obs_dim)

    def reset(self):
        self.agent.reset()

    def act(self, obs):
        one_hot, _ = self.agent.act(obs)
        action = int(np.argmax(one_hot))
        stats = self.agent.last_stats or {}
        probs = np.asarray(
            stats.get("visit_distribution", np.eye(self.action_dim)[action]),
            dtype=np.float64,
        )
        return PolicyOutput(
            action,
            probs,
            float(probs[action]),
            stats,
        )

    def observe_executed(self, action):
        self.agent.observe_executed(action)
