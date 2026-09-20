"""
DreamerV3 Agent for Trading

This implements the full DreamerV3 algorithm:
1. World Model Learning (representation + dynamics + reward prediction)
2. Behavior Learning (actor-critic in imagination)

Based on: https://arxiv.org/abs/2301.04104
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam  # Use Adam instead of AdamW to avoid transformers import issue
import numpy as np

from models.dreamer_components import (
    Encoder, RSSM, Decoder, RewardPredictor, Actor, Critic,
    symlog, symexp
)


class ReplayBuffer:
    """
    Ring buffer of transitions backed by preallocated numpy arrays.

    Sequences are gathered with a single vectorized fancy-index, so sampling cost
    is independent of Python loop overhead (the old deque/dict version took
    seconds per batch at large batch sizes and starved the GPU).
    """
    def __init__(self, capacity=100_000, seq_len=64):
        self.capacity = int(capacity)
        self.seq_len = int(seq_len)
        self.obs = None
        self.action = None
        self.reward = np.zeros(self.capacity, dtype=np.float32)
        self.done = np.zeros(self.capacity, dtype=np.float32)
        self.ptr = 0   # next physical write index
        self.size = 0  # number of valid transitions

    def _allocate(self, obs, action):
        obs = np.asarray(obs, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        self.obs = np.zeros((self.capacity,) + obs.shape, dtype=np.float32)
        self.action = np.zeros((self.capacity,) + action.shape, dtype=np.float32)

    def add(self, obs, action, reward, done):
        """Add a single transition"""
        if self.obs is None:
            self._allocate(obs, action)
        self.obs[self.ptr] = obs
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.done[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """
        Sample batch_size sequences of length seq_len as numpy arrays.
        Returns dict with obs (B,T,obs_dim), action (B,T,A), reward (B,T), done (B,T).
        """
        if self.size < self.seq_len + 1:
            return None

        # Logical index 0 is the oldest transition
        oldest = (self.ptr - self.size) % self.capacity
        starts = np.random.randint(0, self.size - self.seq_len, size=batch_size)
        idx = (oldest + starts[:, None] + np.arange(self.seq_len)[None, :]) % self.capacity

        return {
            'obs': self.obs[idx],
            'action': self.action[idx],
            'reward': self.reward[idx],
            'done': self.done[idx],
        }

    def __len__(self):
        return self.size


class DreamerV3Agent:
    """
    DreamerV3 Agent for Trading

    The agent learns a World Model of the market, then uses it to
    imagine trajectories and improve its policy.
    """
    def __init__(
        self,
        obs_dim,
        action_dim=3,  # [flat, long, short] or [flat, long] for long-only
        device='cpu',
        # Architecture
        embed_dim=256,
        hidden_dim=512,
        stoch_dim=32,
        num_categories=32,
        # Optimization
        lr_world_model=3e-4,
        lr_actor=1e-4,
        lr_critic=3e-4,
        # Hyperparameters
        gamma=0.99,
        lambda_=0.95,  # GAE lambda
        horizon=15,  # imagination horizon
        free_nats=1.0,
        kl_dyn_scale=0.5,
        kl_rep_scale=0.1,
        entropy_coef=3e-4,
        slow_critic_tau=0.02,
        max_imag_starts=8192,  # cap on imagination rollouts per train step
        use_amp=None,  # bf16 autocast; defaults to True on CUDA
    ):
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lambda_ = lambda_
        self.horizon = horizon
        self.use_amp = (str(device).startswith('cuda')) if use_amp is None else bool(use_amp)

        # Build networks
        self.encoder = Encoder(obs_dim, embed_dim).to(device)
        self.rssm = RSSM(embed_dim, hidden_dim, stoch_dim, num_categories, action_dim).to(device)
        self.decoder = Decoder(hidden_dim + stoch_dim * num_categories, obs_dim).to(device)
        self.reward_predictor = RewardPredictor(hidden_dim + stoch_dim * num_categories).to(device)
        self.actor = Actor(hidden_dim + stoch_dim * num_categories, action_dim).to(device)
        self.critic = Critic(hidden_dim + stoch_dim * num_categories).to(device)
        self.slow_critic = copy.deepcopy(self.critic).requires_grad_(False)

        # Optimizers
        self.world_model_params = (
            list(self.encoder.parameters()) +
            list(self.rssm.parameters()) +
            list(self.decoder.parameters()) +
            list(self.reward_predictor.parameters())
        )
        self.optimizer_world_model = Adam(self.world_model_params, lr=lr_world_model)
        self.optimizer_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = Adam(self.critic.parameters(), lr=lr_critic)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=100_000, seq_len=64)

        # Hyperparameters
        self.free_nats = free_nats
        self.kl_dyn_scale = kl_dyn_scale
        self.kl_rep_scale = kl_rep_scale
        self.entropy_coef = entropy_coef
        self.slow_critic_tau = slow_critic_tau
        self.max_imag_starts = max_imag_starts

        # Return normalization (EMA of 5th/95th percentile of lambda-returns)
        self.return_low = None
        self.return_high = None

        # Tracking
        self.training_step = 0
        self.prev_action = None

    def _autocast(self):
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp)

    def act(self, obs, h=None, z=None, deterministic=False):
        """
        Select action given observation

        Args:
            obs: observation (np.array)
            h, z: previous latent state (None if first step)
            deterministic: use greedy action

        Returns:
            action (np.array), new (h, z)
        """
        with torch.no_grad():
            obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device).unsqueeze(0)

            # Encode observation
            embed = self.encoder(obs_t)

            # Initialize or update latent state
            if h is None or z is None or self.prev_action is None:
                h, z = self.rssm.initial_state(1, self.device)
                action = torch.zeros(1, self.action_dim, device=self.device)
            else:
                action = self.prev_action

            # Update latent state
            h, z, _, _ = self.rssm.observe(embed, action, h, z)

            # Get state
            state = self.rssm.get_state(h, z)

            # Sample action
            action = self.actor.sample(state, deterministic=deterministic)

            # Store for next step
            self.prev_action = action

            return action.cpu().numpy()[0], (h, z)

    def train_step(self, batch_size=16):
        """
        Single training step

        1. Train World Model (representation + dynamics + reward)
        2. Imagine trajectories in learned world model
        3. Train Actor-Critic on imagined trajectories
        """
        # Sample batch
        batch = self.replay_buffer.sample(batch_size)
        if batch is None:
            return None

        obs = torch.as_tensor(batch['obs'], device=self.device)  # (B, T, obs_dim)
        action = torch.as_tensor(batch['action'], device=self.device)  # (B, T, action_dim)
        reward = torch.as_tensor(batch['reward'], device=self.device)  # (B, T)
        done = torch.as_tensor(batch['done'], device=self.device)  # (B, T)

        B, T = obs.shape[0], obs.shape[1]

        # ==================== PHASE 1: Train World Model ====================
        self.optimizer_world_model.zero_grad(set_to_none=True)

        with self._autocast():
            # Encode observations
            embed = self.encoder(obs.reshape(B * T, -1)).reshape(B, T, -1)

            # Initialize state
            h, z = self.rssm.initial_state(B, self.device)

            states = []
            kl_losses = []

            # Unroll sequence
            for t in range(T):
                if t > 0:
                    # Reset latent state at episode boundaries
                    keep = (1.0 - done[:, t - 1]).unsqueeze(-1)
                    h = h * keep
                    z = z * keep

                # Posterior inference
                h, z, prior_logits, posterior_logits = self.rssm.observe(
                    embed[:, t], action[:, t], h, z
                )
                states.append(self.rssm.get_state(h, z))
                kl_losses.append(
                    self.rssm.kl_loss(prior_logits, posterior_logits,
                                      self.free_nats, self.kl_dyn_scale, self.kl_rep_scale)
                )

            states_seq = torch.stack(states, dim=1)  # (B, T, state_dim)
            flat_states = states_seq.reshape(B * T, -1)

            # Reconstruction loss in symlog space (sum over obs dims, mean over batch/time)
            obs_pred = self.decoder(flat_states).float()
            recon_loss_total = ((obs_pred - symlog(obs.reshape(B * T, -1))) ** 2).sum(-1).mean()

            # Reward prediction loss (symlog space)
            reward_pred = self.reward_predictor(flat_states).float()
            reward_loss_total = F.mse_loss(reward_pred, symlog(reward.reshape(B * T)))

            kl_loss_total = torch.stack(kl_losses).mean()

            world_model_loss = recon_loss_total + reward_loss_total + kl_loss_total

        world_model_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model_params, max_norm=1000.0)
        self.optimizer_world_model.step()

        # ==================== PHASE 2: Imagine & Train Actor-Critic ====================

        # Start imagination from the posterior states of the batch
        with torch.no_grad():
            start_states = states_seq.detach().reshape(B * T, -1)
            if start_states.shape[0] > self.max_imag_starts:
                pick = torch.randperm(start_states.shape[0], device=self.device)[:self.max_imag_starts]
                start_states = start_states[pick]
            h_start = start_states[:, :self.rssm.hidden_dim].contiguous()
            z_start = start_states[:, self.rssm.hidden_dim:].contiguous()

            imag_states, imag_actions, imag_rewards = self._imagine_trajectory(
                h_start, z_start, self.horizon
            )
            # imag_states: (N, H+1, S), imag_actions: (N, H), imag_rewards: (N, H+1)

            with self._autocast():
                slow_values = symexp(self.slow_critic(imag_states.reshape(-1, imag_states.shape[-1])).float())
            slow_values = slow_values.reshape(imag_states.shape[0], self.horizon + 1)

            returns = self._lambda_returns(imag_rewards[:, 1:], slow_values)  # (N, H)
            advantages = self._normalize_returns(returns) - self._normalize_returns(slow_values[:, :-1])

        actor_states = imag_states[:, :-1].reshape(-1, imag_states.shape[-1])  # (N*H, S)

        # Train critic: regress toward lambda-returns (symlog space)
        self.optimizer_critic.zero_grad(set_to_none=True)
        with self._autocast():
            value_pred = self.critic(actor_states).float()
            value_loss = F.mse_loss(value_pred, symlog(returns.reshape(-1)))
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=100.0)
        self.optimizer_critic.step()
        self._update_slow_critic()

        # Train actor: REINFORCE on the actions actually taken in imagination + entropy bonus
        self.optimizer_actor.zero_grad(set_to_none=True)
        with self._autocast():
            dist = self.actor.dist(actor_states)
            log_probs = dist.log_prob(imag_actions.reshape(-1))
            entropy = dist.entropy()
            policy_loss = -(log_probs * advantages.reshape(-1)).mean() - self.entropy_coef * entropy.mean()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=100.0)
        self.optimizer_actor.step()

        self.training_step += 1

        return {
            'world_model_loss': world_model_loss.item(),
            'recon_loss': recon_loss_total.item(),
            'reward_loss': reward_loss_total.item(),
            'kl_loss': kl_loss_total.item(),
            'value_loss': value_loss.item(),
            'policy_loss': policy_loss.item(),
            'entropy': entropy.mean().item(),
        }

    def _imagine_trajectory(self, h, z, horizon):
        """
        Roll out the policy inside the world model (no gradients).

        Returns:
            states: (N, H+1, state_dim)  latent states s_0..s_H
            actions: (N, H)              action indices taken at s_0..s_{H-1}
            rewards: (N, H+1)            predicted reward for arriving in s_0..s_H
        """
        states, actions, rewards = [], [], []

        with self._autocast():
            for t in range(horizon + 1):
                state = self.rssm.get_state(h, z)
                states.append(state)
                rewards.append(symexp(self.reward_predictor(state).float()))
                if t == horizon:
                    break
                action_idx = self.actor.dist(state).sample()
                actions.append(action_idx)
                action = F.one_hot(action_idx, self.action_dim).float()
                h, z, _ = self.rssm.imagine(action, h, z)

        return (
            torch.stack(states, dim=1).float(),
            torch.stack(actions, dim=1),
            torch.stack(rewards, dim=1),
        )

    def _lambda_returns(self, rewards, values):
        """
        R_t = r_{t+1} + gamma * ((1 - lambda) * v_{t+1} + lambda * R_{t+1}), R_H = v_H

        Args:
            rewards: (N, H)   reward received when moving s_t -> s_{t+1}
            values:  (N, H+1) bootstrap values for s_0..s_H
        Returns:
            (N, H) lambda-returns for s_0..s_{H-1}
        """
        H = rewards.shape[1]
        returns = torch.zeros_like(rewards)
        next_return = values[:, -1]
        for t in reversed(range(H)):
            next_return = rewards[:, t] + self.gamma * (
                (1 - self.lambda_) * values[:, t + 1] + self.lambda_ * next_return
            )
            returns[:, t] = next_return
        return returns

    def _normalize_returns(self, x):
        """Scale by EMA of the 5-95 percentile range of returns (DreamerV3), clipped below at 1."""
        if self.return_low is None:
            self.return_low = torch.quantile(x.detach().flatten(), 0.05)
            self.return_high = torch.quantile(x.detach().flatten(), 0.95)
        else:
            decay = 0.99
            self.return_low = decay * self.return_low + (1 - decay) * torch.quantile(x.detach().flatten(), 0.05)
            self.return_high = decay * self.return_high + (1 - decay) * torch.quantile(x.detach().flatten(), 0.95)
        scale = torch.clamp(self.return_high - self.return_low, min=1.0)
        return x / scale

    def _update_slow_critic(self):
        with torch.no_grad():
            for p_slow, p in zip(self.slow_critic.parameters(), self.critic.parameters()):
                p_slow.lerp_(p, self.slow_critic_tau)

    def save(self, path):
        """Save agent"""
        torch.save({
            'encoder': self.encoder.state_dict(),
            'rssm': self.rssm.state_dict(),
            'decoder': self.decoder.state_dict(),
            'reward_predictor': self.reward_predictor.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'slow_critic': self.slow_critic.state_dict(),
            'training_step': self.training_step,
        }, path)

    def load(self, path):
        """Load agent"""
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.rssm.load_state_dict(checkpoint['rssm'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.reward_predictor.load_state_dict(checkpoint['reward_predictor'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.slow_critic.load_state_dict(checkpoint.get('slow_critic', checkpoint['critic']))
        self.training_step = checkpoint.get('training_step', 0)
        print(f"Loaded checkpoint from step {self.training_step}")


if __name__ == "__main__":
    # Quick test
    print("Testing DreamerV3 Agent...")

    obs_dim = 64 * 11  # window * features
    agent = DreamerV3Agent(obs_dim, action_dim=3, device='cpu')

    # Test act
    obs = np.random.randn(obs_dim)
    action, (h, z) = agent.act(obs)
    print(f"✅ Action: {action.shape}")

    # Test training
    # Add some dummy data
    for _ in range(100):
        obs = np.random.randn(obs_dim)
        action = np.random.randn(3)
        reward = np.random.randn()
        done = False
        agent.replay_buffer.add(obs, action, reward, done)

    # Train step
    losses = agent.train_step(batch_size=4)
    if losses:
        print(f"✅ Training step completed")
        print(f"   World Model Loss: {losses['world_model_loss']:.4f}")
        print(f"   Value Loss: {losses['value_loss']:.4f}")
        print(f"   Policy Loss: {losses['policy_loss']:.4f}")

    print("\n🎉 DreamerV3 Agent working!")
