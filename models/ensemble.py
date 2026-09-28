from collections import Counter

import numpy as np
import torch

from models.policy import PolicyOutput


class EnsemblePolicy:
    def __init__(self, members, min_agreement=0.6, vote="soft", weights=None):
        if not members:
            raise ValueError("an ensemble requires at least one member")
        if vote not in {"soft", "hard"}:
            raise ValueError("vote must be 'soft' or 'hard'")
        self.members = list(members)
        self.min_agreement = float(min_agreement)
        if not 0.0 <= self.min_agreement <= 1.0:
            raise ValueError("min_agreement must be between 0 and 1")
        self.vote = vote
        self.action_dim = max(int(member.action_dim) for member in self.members)
        if self.action_dim < 2:
            raise ValueError("ensemble members must have at least two actions")
        self.obs_dim = int(self.members[0].obs_dim)
        if any(int(member.obs_dim) != self.obs_dim for member in self.members):
            raise ValueError("ensemble members must have the same obs_dim")
        if any(int(member.action_dim) > 3 for member in self.members):
            raise ValueError("ensemble supports at most three discrete actions")
        raw_weights = np.ones(len(self.members), dtype=np.float64) if weights is None else np.asarray(
            weights, dtype=np.float64
        )
        if raw_weights.shape != (len(self.members),):
            raise ValueError("weights must contain one value per ensemble member")
        if not np.isfinite(raw_weights).all() or np.any(raw_weights < 0):
            raise ValueError("weights must be finite and non-negative")
        if raw_weights.sum() <= 0:
            raise ValueError("at least one ensemble weight must be positive")
        self.weights = raw_weights / raw_weights.sum()

    def reset(self):
        for member in self.members:
            member.reset()

    def observe_executed(self, action):
        for member in self.members:
            member_action = int(action)
            if member_action >= int(member.action_dim):
                member_action = 0
            member.observe_executed(member_action)

    def _member_prediction(self, member, obs):
        output = member.act(obs)
        if isinstance(output, PolicyOutput):
            action = int(output.action)
            probs = np.asarray(output.probs, dtype=np.float64).reshape(-1)
        else:
            if isinstance(output, tuple):
                output = output[0]
            values = np.asarray(output)
            if values.ndim == 0 or values.size == 1:
                action = int(values.reshape(-1)[0])
                probs = np.zeros(int(member.action_dim), dtype=np.float64)
                probs[action] = 1.0
            else:
                probs = values.astype(np.float64).reshape(-1)
                action = int(np.argmax(probs))
        member_dim = int(member.action_dim)
        if not 0 <= action < member_dim:
            raise ValueError(f"member returned invalid action {action}")
        if probs.size != member_dim:
            raise ValueError(
                f"member probability width {probs.size} != action_dim {member_dim}"
            )
        if not np.isfinite(probs).all() or np.any(probs < 0) or probs.sum() <= 0:
            raise ValueError("member returned invalid action probabilities")
        probs = probs / probs.sum()
        padded = np.zeros(self.action_dim, dtype=np.float64)
        padded[:member_dim] = probs
        return action, padded

    def act(self, obs):
        predictions = [
            self._member_prediction(member, obs) for member in self.members
        ]
        member_actions = [action for action, _ in predictions]
        member_probs = np.stack([probs for _, probs in predictions])
        if self.vote == "soft":
            probs = np.average(member_probs, axis=0, weights=self.weights)
            selected = int(np.argmax(probs))
        else:
            counts = np.zeros(self.action_dim, dtype=np.float64)
            for weight, action in zip(self.weights, member_actions):
                counts[action] += weight
            probs = counts
            probs /= probs.sum()
            selected = int(np.argmax(counts))

        agreement = float(
            sum(weight for weight, action in zip(self.weights, member_actions) if action == selected)
        )
        consensus = agreement >= self.min_agreement
        action = selected if consensus else 0
        entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-12))))
        uncertainty = entropy / np.log(self.action_dim) if self.action_dim > 1 else 0.0
        kl = np.sum(
            member_probs
            * np.log(np.maximum(member_probs, 1e-12) / np.maximum(probs, 1e-12)),
            axis=1,
        )
        epistemic = float(np.average(kl, weights=self.weights))
        return PolicyOutput(
            action=action,
            probs=probs,
            confidence=float(probs[action]),
            info={
                "member_actions": member_actions,
                "agreement": agreement,
                "consensus": consensus,
                "uncertainty": float(uncertainty),
                "epistemic": epistemic,
                "epistemic_uncertainty": epistemic,
                "epistemic_disagreement": epistemic,
            },
        )


class EnsembleAgent:
    def __init__(self, agent_class, num_models=5, **agent_kwargs):
        self.num_models = int(num_models)
        if self.num_models < 1:
            raise ValueError("num_models must be positive")
        self.agent_kwargs = agent_kwargs.copy()
        self.models = []
        for index in range(self.num_models):
            kwargs = agent_kwargs.copy()
            if "hidden_dim" in kwargs:
                kwargs["hidden_dim"] += index * 16
            torch.manual_seed(42 + index)
            np.random.seed(42 + index)
            self.models.append(agent_class(**kwargs))
        self._members = None

    @staticmethod
    def _action(result):
        if isinstance(result, tuple):
            result = result[0]
        values = np.asarray(result)
        if values.ndim == 0 or values.size == 1:
            return int(values.reshape(-1)[0])
        return int(np.argmax(values))

    @staticmethod
    def _member_action_dim(model, result, configured_action_dim=None):
        action_dim = getattr(model, "action_dim", configured_action_dim)
        if action_dim is not None:
            return int(action_dim)
        value = result[0] if isinstance(result, tuple) else result
        if isinstance(value, PolicyOutput):
            return len(value.probs)
        values = np.asarray(value)
        if values.ndim == 0 or values.size == 1:
            return max(int(values.reshape(-1)[0]) + 1, 2)
        return int(values.size)

    def _build_members(self, obs, results):
        obs_dim = int(np.asarray(obs).size)
        configured_action_dim = self.agent_kwargs.get("action_dim")
        self._members = [
            _LegacyMemberPolicy(
                model,
                self._member_action_dim(model, result, configured_action_dim),
                int(getattr(model, "obs_dim", obs_dim)),
                pending_result=result,
            )
            for model, result in zip(self.models, results)
        ]

    def act(self, obs, use_consensus=True, consensus_threshold=3):
        results = [model.act(obs) for model in self.models]
        if self._members is None:
            self._build_members(obs, results)
        else:
            for member, result in zip(self._members, results):
                member.pending_result = result
        threshold = float(consensus_threshold)
        min_agreement = (
            min(max(threshold / self.num_models, 0.0), 1.0)
            if use_consensus
            else 0.0
        )
        policy = EnsemblePolicy(
            self._members,
            min_agreement=min_agreement,
            vote="hard",
        )
        output = policy.act(obs)
        actions = output.info["member_actions"]
        counts = Counter(actions)
        majority_action, majority_count = counts.most_common(1)[0]
        consensus = output.info["consensus"] and not (
            use_consensus and threshold > self.num_models
        )
        action = output.action if not use_consensus or consensus else 0
        return action, {
            "actions": actions,
            "action_counts": dict(counts),
            "majority_action": majority_action,
            "majority_count": majority_count,
            **output.info,
            "consensus": consensus if use_consensus else True,
            "q_values": None,
        }

    def reset(self):
        for model in self.models:
            reset = getattr(model, "reset", None)
            if reset is not None:
                reset()

    def observe_executed(self, action):
        for member in self._members or ():
            member.observe_executed(action)

    def get_uncertainty(self, actions):
        counts = Counter(actions)
        probabilities = np.asarray(list(counts.values()), dtype=np.float64) / len(actions)
        return -float(np.sum(probabilities * np.log(probabilities + 1e-10)))

    def train(self, *args, **kwargs):
        for model in self.models:
            model.train(*args, **kwargs)

    def save(self, path_prefix):
        for index, model in enumerate(self.models):
            model.save(f"{path_prefix}_model{index}.pt")

    def load(self, path_prefix):
        for index, model in enumerate(self.models):
            model.load(f"{path_prefix}_model{index}.pt")


class _LegacyMemberPolicy:
    def __init__(self, model, action_dim, obs_dim, pending_result=None):
        self.model = model
        self.action_dim = int(action_dim)
        self.obs_dim = int(obs_dim)
        self.pending_result = pending_result

    def act(self, obs):
        if self.pending_result is not None:
            result = self.pending_result
            self.pending_result = None
            return result
        return self.model.act(obs)

    def reset(self):
        reset = getattr(self.model, "reset", None)
        if reset is not None:
            reset()

    def observe_executed(self, action):
        observe = getattr(self.model, "observe_executed", None)
        if observe is not None:
            member_action = int(action)
            if member_action >= self.action_dim:
                member_action = 0
            observe(member_action)
