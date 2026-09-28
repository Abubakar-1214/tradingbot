import math

import numpy as np
import torch
import torch.nn.functional as F

from models.dreamer_components import symexp


class MinMaxStats:
    def __init__(self):
        self.minimum = float("inf")
        self.maximum = -float("inf")

    def update(self, value):
        self.minimum = min(self.minimum, float(value))
        self.maximum = max(self.maximum, float(value))

    def normalize(self, value):
        if self.minimum == float("inf") or self.maximum <= self.minimum:
            return 0.0
        return (float(value) - self.minimum) / (self.maximum - self.minimum)


class MCTSNode:
    def __init__(self, h, z, prior=0.0, parent=None, action=None, reward=0.0):
        self.h = h
        self.z = z
        self.prior = float(prior)
        self.parent = parent
        self.action = action
        self.reward = float(reward)
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0

    @property
    def value(self):
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    @property
    def predicted_reward(self):
        return self.reward

    def expanded(self):
        return bool(self.children)

    def select_child(self, c_puct, min_max_stats):
        best_action = None
        best_child = None
        best_score = -float("inf")
        exploration_scale = math.sqrt(max(self.visit_count, 1))
        for action, child in self.children.items():
            q_value = min_max_stats.normalize(child.value) if child.visit_count else 0.0
            u_value = (
                c_puct
                * child.prior
                * exploration_scale
                / (1 + child.visit_count)
            )
            score = q_value + u_value
            if score > best_score:
                best_action = action
                best_child = child
                best_score = score
        return best_action, best_child

    def backup(self, value):
        self.visit_count += 1
        self.value_sum += float(value)


class MCTS:
    def __init__(
        self,
        agent,
        num_simulations=100,
        c_puct=1.0,
        gamma=0.99,
        dirichlet_alpha=0.3,
        dirichlet_fraction=0.25,
        root_noise=False,
        seed=None,
    ):
        self.agent = agent
        self.num_simulations = int(num_simulations)
        self.c_puct = float(c_puct)
        self.gamma = float(gamma)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.dirichlet_fraction = float(dirichlet_fraction)
        self.root_noise = bool(root_noise)
        self.rng = np.random.default_rng(seed)
        if self.num_simulations < 1:
            raise ValueError("num_simulations must be positive")
        if self.c_puct < 0 or not 0 <= self.dirichlet_fraction <= 1:
            raise ValueError("invalid PUCT or Dirichlet configuration")
        self.action_dim = int(agent.action_dim)
        self.actions = np.eye(self.action_dim, dtype=np.float32)
        self.min_max_stats = MinMaxStats()
        self.root = None

    def _set_eval(self):
        for name in ("encoder", "rssm", "actor", "critic", "reward_predictor"):
            network = getattr(self.agent, name, None)
            if network is not None:
                network.eval()

    def _expand(self, node):
        with torch.no_grad():
            state = self.agent.rssm.get_state(node.h, node.z)
            priors = torch.softmax(self.agent.actor(state), dim=-1)[0].cpu().numpy()
            for action_idx, action in enumerate(self.actions):
                action_tensor = torch.as_tensor(
                    action, dtype=torch.float32, device=self.agent.device
                ).unsqueeze(0)
                h_next, z_next, _ = self.agent.rssm.imagine(action_tensor, node.h, node.z)
                next_state = self.agent.rssm.get_state(h_next, z_next)
                reward = float(symexp(self.agent.reward_predictor(next_state).float()).item())
                node.children[action_idx] = MCTSNode(
                    h_next,
                    z_next,
                    prior=priors[action_idx],
                    parent=node,
                    action=action_idx,
                    reward=reward,
                )

    def _add_root_noise(self, root):
        if not self.root_noise or self.dirichlet_fraction == 0:
            return
        noise = self.rng.dirichlet(
            np.full(self.action_dim, self.dirichlet_alpha, dtype=np.float64)
        )
        for action_idx, child in root.children.items():
            child.prior = (
                (1.0 - self.dirichlet_fraction) * child.prior
                + self.dirichlet_fraction * float(noise[action_idx])
            )

    def backup(self, path, leaf_value):
        value = float(leaf_value)
        for node in reversed(path):
            if node.parent is not None:
                value = node.reward + self.gamma * value
            node.backup(value)
            if node.parent is not None:
                self.min_max_stats.update(node.value)

    def search(self, h, z):
        self._set_eval()
        self.min_max_stats = MinMaxStats()
        root = MCTSNode(h, z)
        self._expand(root)
        self._add_root_noise(root)
        for _ in range(self.num_simulations):
            node = root
            path = [root]
            while node.expanded():
                _, node = node.select_child(self.c_puct, self.min_max_stats)
                path.append(node)
                if not node.expanded():
                    break
            if not node.expanded():
                self._expand(node)
            with torch.no_grad():
                state = self.agent.rssm.get_state(node.h, node.z)
                value = float(symexp(self.agent.critic(state).float()).item())
            self.backup(path, value)

        visit_counts = {
            action: child.visit_count for action, child in root.children.items()
        }
        total_visits = sum(visit_counts.values())
        visit_distribution = np.asarray(
            [visit_counts[index] for index in range(self.action_dim)],
            dtype=np.float64,
        )
        if total_visits:
            visit_distribution /= total_visits
        q_values = {
            action: child.value for action, child in root.children.items()
        }
        visit_counts_list = [visit_counts[index] for index in range(self.action_dim)]
        q_values_list = [q_values[index] for index in range(self.action_dim)]
        action = max(range(self.action_dim), key=lambda index: visit_counts[index])
        stats = {
            "visit_counts": visit_counts_list,
            "q_values": q_values_list,
            "probs": visit_distribution.tolist(),
            "visit_counts_by_action": visit_counts,
            "q_values_by_action": q_values,
            "visit_distribution": visit_distribution,
            "root_visits": root.visit_count,
        }
        self.root = root
        return action, stats

    def search_with_stats(self, h, z):
        return self.search(h, z)


class DreamerMCTSAgent:
    def __init__(
        self,
        dreamer_agent,
        num_simulations=50,
        c_puct=1.0,
        **mcts_kwargs,
    ):
        self.dreamer = dreamer_agent
        self.mcts = MCTS(
            dreamer_agent,
            num_simulations=num_simulations,
            c_puct=c_puct,
            **mcts_kwargs,
        )
        self.reset()

    def reset(self):
        self.h = None
        self.z = None
        self.prev_action = None
        self.last_stats = None
        self.mcts._set_eval()

    def act(self, obs, h=None, z=None, use_mcts=True):
        with torch.no_grad():
            obs_t = torch.as_tensor(
                np.asarray(obs, dtype=np.float32), device=self.dreamer.device
            ).reshape(1, -1)
            embed = self.dreamer.encoder(obs_t)
            if h is None or z is None:
                h = self.h
                z = self.z
            if h is None or z is None:
                h, z = self.dreamer.rssm.initial_state(1, self.dreamer.device)
                previous_action = torch.zeros(
                    1, self.dreamer.action_dim, device=self.dreamer.device
                )
            else:
                previous_action = self.prev_action
                if previous_action is None:
                    previous_action = torch.zeros(
                        1, self.dreamer.action_dim, device=self.dreamer.device
                    )
            h, z, _, _ = self.dreamer.rssm.observe(
                embed, previous_action, h, z
            )
            if use_mcts:
                action_idx, self.last_stats = self.mcts.search(h, z)
                action = F.one_hot(
                    torch.tensor([action_idx], device=self.dreamer.device),
                    self.dreamer.action_dim,
                ).float()
            else:
                state = self.dreamer.rssm.get_state(h, z)
                action = self.dreamer.actor.sample(state, deterministic=True)
                action_idx = int(action.argmax(dim=-1).item())
                probabilities = np.eye(self.dreamer.action_dim)[action_idx].tolist()
                self.last_stats = {
                    "visit_counts": [0] * self.dreamer.action_dim,
                    "q_values": [0.0] * self.dreamer.action_dim,
                    "probs": probabilities,
                    "visit_counts_by_action": {},
                    "q_values_by_action": {},
                    "visit_distribution": np.eye(self.dreamer.action_dim)[action_idx],
                    "root_visits": 0,
                }
            self.h = h
            self.z = z
            self.prev_action = action
            return action.cpu().numpy()[0], (h, z)

    def observe_executed(self, action):
        if not 0 <= int(action) < self.dreamer.action_dim:
            raise ValueError(f"action {action} outside [0, {self.dreamer.action_dim})")
        self.prev_action = F.one_hot(
            torch.tensor([int(action)], device=self.dreamer.device),
            self.dreamer.action_dim,
        ).float()
