import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical


def tokenize_observation(observation, window, n_features):
    observation = torch.as_tensor(observation, dtype=torch.float32)
    single = observation.ndim == 1
    if single:
        observation = observation.unsqueeze(0)
    if observation.ndim == 3:
        expected = n_features + 5
        if observation.shape[1:] != (window, expected):
            raise ValueError(
                f"expected token shape (*, {window}, {expected}), got {tuple(observation.shape)}"
            )
        return observation.squeeze(0) if single else observation
    if observation.ndim != 2:
        raise ValueError("observation must be flat, batched-flat, or tokenized")
    expected = window * n_features + 5
    if observation.shape[-1] != expected:
        raise ValueError(f"expected observation width {expected}, got {observation.shape[-1]}")
    features = observation[:, :window * n_features].reshape(
        -1, window, n_features
    )
    account = observation[:, window * n_features:].unsqueeze(1).expand(-1, window, -1)
    tokens = torch.cat((features, account), dim=-1)
    return tokens.squeeze(0) if single else tokens


def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    values = torch.as_tensor(values, dtype=torch.float32)
    rewards = torch.as_tensor(rewards, dtype=torch.float32, device=values.device)
    dones = torch.as_tensor(dones, dtype=torch.float32, device=values.device)
    last_value = torch.as_tensor(last_value, dtype=torch.float32, device=values.device)
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(last_value)
    for t in reversed(range(len(rewards))):
        next_value = last_value if t == len(rewards) - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        advantages[t] = gae
    return advantages, advantages + values


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(positions * divisor)
        if d_model > 1:
            encoding[:, 1::2] = torch.cos(positions * divisor[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding)

    def forward(self, inputs):
        if inputs.shape[1] > len(self.encoding):
            raise ValueError("sequence exceeds configured maximum length")
        return inputs + self.encoding[:inputs.shape[1]].unsqueeze(0)


class _TransformerEncoder(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_heads, num_layers, seq_len, dropout):
        super().__init__()
        self.embedding = nn.Linear(state_dim, hidden_dim)
        self.position = PositionalEncoding(hidden_dim, seq_len)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, sequence):
        x = self.position(self.embedding(sequence))
        return self.transformer(x)

    def attention_weights(self, sequence):
        x = self.position(self.embedding(sequence))
        weights = []
        for layer in self.transformer.layers:
            if layer.norm_first:
                query = layer.norm1(x)
                attended, layer_weights = layer.self_attn(
                    query,
                    query,
                    query,
                    need_weights=True,
                    average_attn_weights=True,
                )
                x = x + layer.dropout1(attended)
                x = x + layer._ff_block(layer.norm2(x))
            else:
                attended, layer_weights = layer.self_attn(
                    x,
                    x,
                    x,
                    need_weights=True,
                    average_attn_weights=True,
                )
                x = layer.norm1(x + layer.dropout1(attended))
                x = layer.norm2(x + layer._ff_block(x))
            weights.append(layer_weights)
        return torch.stack(weights)


class TransformerActor(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        seq_len=64,
        dropout=0.0,
    ):
        super().__init__()
        self.encoder = _TransformerEncoder(
            state_dim, hidden_dim, num_heads, num_layers, seq_len, dropout
        )
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, state_sequence):
        return self.action_head(self.encoder(state_sequence)[:, -1])

    def get_attention_weights(self, state_sequence):
        return self.encoder.attention_weights(state_sequence)


class TransformerCritic(nn.Module):
    def __init__(
        self,
        state_dim,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        seq_len=64,
        dropout=0.0,
    ):
        super().__init__()
        self.encoder = _TransformerEncoder(
            state_dim, hidden_dim, num_heads, num_layers, seq_len, dropout
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state_sequence):
        return self.value_head(self.encoder(state_sequence)[:, -1]).squeeze(-1)


class TransformerAgentWrapper:
    def __init__(
        self,
        state_dim=None,
        action_dim=3,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        seq_len=64,
        *,
        window=None,
        n_features=None,
        obs_dim=None,
        device="cpu",
        learning_rate=3e-4,
        clip_coef=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        update_epochs=4,
        num_minibatches=4,
        dropout=0.0,
    ):
        self.window = int(window if window is not None else seq_len)
        if n_features is None:
            if state_dim is not None:
                n_features = int(state_dim) - 5
            elif obs_dim is not None:
                n_features = (int(obs_dim) - 5) // self.window
            else:
                raise ValueError("state_dim, n_features, or obs_dim is required")
        self.n_features = int(n_features)
        self.state_dim = self.n_features + 5
        self.obs_dim = self.window * self.n_features + 5
        if obs_dim is not None and int(obs_dim) != self.obs_dim:
            raise ValueError(f"obs_dim {obs_dim} does not match token dimensions {self.obs_dim}")
        self.action_dim = int(action_dim)
        self.seq_len = self.window
        self.device = torch.device(device)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.clip_coef = float(clip_coef)
        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.update_epochs = int(update_epochs)
        self.num_minibatches = int(num_minibatches)
        self.training_step = 0

        self.actor = TransformerActor(
            self.state_dim,
            self.action_dim,
            self.hidden_dim,
            self.num_heads,
            self.num_layers,
            self.seq_len,
            self.dropout,
        ).to(self.device)
        self.critic = TransformerCritic(
            self.state_dim,
            self.hidden_dim,
            self.num_heads,
            self.num_layers,
            self.seq_len,
            self.dropout,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=learning_rate,
        )
        self.learning_rate = float(learning_rate)
        self.config = {
            "n_features": self.n_features,
            "window": self.window,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "device": str(self.device),
            "learning_rate": self.learning_rate,
            "clip_coef": self.clip_coef,
            "value_coef": self.value_coef,
            "entropy_coef": self.entropy_coef,
            "max_grad_norm": self.max_grad_norm,
            "update_epochs": self.update_epochs,
            "num_minibatches": self.num_minibatches,
            "dropout": self.dropout,
        }

    def _tokens(self, observations):
        tokens = tokenize_observation(
            observations, self.window, self.n_features
        ).to(self.device)
        return tokens.unsqueeze(0) if tokens.ndim == 2 else tokens

    def evaluate_actions(self, obs_batch, actions):
        tokens = self._tokens(obs_batch)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device).reshape(-1)
        logits = self.actor(tokens)
        distribution = Categorical(logits=logits)
        return distribution.log_prob(actions), distribution.entropy(), self.critic(tokens)

    def policy_probs(self, obs_batch):
        was_training = self.actor.training
        self.actor.eval()
        with torch.no_grad():
            probs = torch.softmax(self.actor(self._tokens(obs_batch)), dim=-1)
        self.actor.train(was_training)
        return probs

    def act(self, obs, deterministic=True):
        was_actor_training = self.actor.training
        was_critic_training = self.critic.training
        self.actor.eval()
        self.critic.eval()
        tokens = self._tokens(obs)
        with torch.no_grad():
            logits = self.actor(tokens)
            distribution = Categorical(logits=logits)
            action = logits.argmax(dim=-1) if deterministic else distribution.sample()
            logp = distribution.log_prob(action)
            value = self.critic(tokens)
        self.actor.train(was_actor_training)
        self.critic.train(was_critic_training)
        return int(action.item()), float(logp.item()), float(value.item())

    def get_attention_weights(self, obs):
        was_training = self.actor.training
        self.actor.eval()
        with torch.no_grad():
            weights = self.actor.get_attention_weights(self._tokens(obs))
        self.actor.train(was_training)
        if weights.shape[1] != 1:
            raise ValueError("get_attention_weights expects a single observation")
        return weights[:, 0].detach().cpu().numpy()

    def train_step(self, batch):
        observations = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device).reshape(-1)
        old_logp = torch.as_tensor(batch["old_logp"], dtype=torch.float32, device=self.device).reshape(-1)
        advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=self.device).reshape(-1)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=self.device).reshape(-1)
        if len(observations) == 0:
            raise ValueError("PPO batch must not be empty")
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        batch_size = len(observations)
        minibatch_size = max(1, math.ceil(batch_size / max(self.num_minibatches, 1)))
        metrics = []
        self.actor.train()
        self.critic.train()
        for _ in range(self.update_epochs):
            permutation = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                indices = permutation[start:start + minibatch_size]
                logp, entropy, values = self.evaluate_actions(
                    observations[indices], actions[indices]
                )
                ratio = torch.exp(logp - old_logp[indices])
                unclipped = ratio * advantages[indices]
                clipped = torch.clamp(
                    ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef
                ) * advantages[indices]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = F.mse_loss(values, returns[indices])
                entropy_mean = entropy.mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_mean
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()
                metrics.append(
                    (
                        float(loss.detach()),
                        float(policy_loss.detach()),
                        float(value_loss.detach()),
                        float(entropy_mean.detach()),
                        float((old_logp[indices] - logp).mean().detach()),
                        float(((ratio - 1.0).abs() > self.clip_coef).float().mean().detach()),
                    )
                )
        self.training_step += 1
        means = np.asarray(metrics, dtype=np.float64).mean(axis=0)
        return {
            "loss": float(means[0]),
            "policy_loss": float(means[1]),
            "value_loss": float(means[2]),
            "entropy": float(means[3]),
            "approx_kl": float(means[4]),
            "clip_fraction": float(means[5]),
        }

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.config,
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "training_step": self.training_step,
            },
            path,
        )

    def load(self, path, map_location=None):
        checkpoint = torch.load(
            path,
            map_location=map_location or self.device,
            weights_only=False,
        )
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.training_step = int(checkpoint.get("training_step", 0))
        return self

    @classmethod
    def from_checkpoint(cls, path, map_location="cpu"):
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        config = dict(checkpoint["config"])
        config["device"] = str(map_location)
        agent = cls(**config)
        agent.load(path, map_location=map_location)
        return agent
