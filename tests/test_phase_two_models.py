import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from core.model_artifacts import (
    ModelArtifactError,
    ModelManifest,
    save_manifest,
)
from env.dreamer_trading_env import RealisticTradingEnv
from models.adversarial_training import (
    MarketMakerAgent,
    SelfPlayTrainer,
)
from models.dreamer_agent import DreamerV3Agent
from models.ensemble import EnsembleAgent, EnsemblePolicy
from models.mcts import MCTS, DreamerMCTSAgent, MCTSNode
from models.meta_learning import (
    MAMLTrader,
    MarketRegimeGenerator,
    build_task_buffer,
)
from models.policy import DreamerMCTSPolicy, PolicyOutput
from models.registry import load_policy
from models.transformer_policy import (
    TransformerAgentWrapper,
    compute_gae,
    tokenize_observation,
)
from tests.helpers import sample_df
from train.data import prepare_data
from train.make_mcts_manifest import make_manifest
from train.train_ensemble import train as train_ensemble
from train.train_transformer import train as train_transformer


def _tiny_dreamer(obs_dim, action_dim=3, seq_len=3):
    return DreamerV3Agent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        embed_dim=4,
        hidden_dim=8,
        stoch_dim=1,
        num_categories=2,
        horizon=1,
        seq_len=seq_len,
        use_amp=False,
        max_imag_starts=16,
    )


def _market_env(
    length=80,
    window=4,
    seed=13,
    features=None,
    returns=None,
    **kwargs,
):
    rng = np.random.default_rng(seed)
    features = (
        rng.normal(size=(length, 2)).astype(np.float32)
        if features is None
        else np.asarray(features, dtype=np.float32)
    )
    returns = (
        rng.normal(0.0, 0.001, size=length).astype(np.float32)
        if returns is None
        else np.asarray(returns, dtype=np.float32)
    )
    return RealisticTradingEnv(
        features,
        returns,
        window=window,
        random_start=False,
        max_episode_steps=None,
        max_drawdown=None,
        seed=seed,
        **kwargs,
    )


def _write_artifact(
    directory,
    model_type,
    model_file,
    contract,
    window,
    n_features,
    action_dim,
    extra=None,
    members=None,
):
    directory.mkdir(parents=True, exist_ok=True)
    contract_file = "feature_contract.json"
    (directory / contract_file).write_text(json.dumps(contract), encoding="utf-8")
    manifest = ModelManifest(
        model_type=model_type,
        model_file=model_file,
        contract_file=contract_file,
        contract_hash=contract["hash"],
        window=window,
        n_features=n_features,
        obs_dim=window * n_features + 5,
        action_dim=action_dim,
        allow_short=action_dim == 3,
        symbol="XAUUSD",
        timeframe="H1",
        train_start="2020-01-01",
        train_end="2022-01-01",
        test_end=None,
        created_at="2024-01-01T00:00:00+00:00",
        git_commit=None,
        hyperparams={},
        members=list(members or []),
        extra=dict(extra or {}),
    )
    return save_manifest(manifest, directory)


def _make_transformer_artifact(
    directory,
    contract,
    window=3,
    n_features=2,
    action_dim=3,
):
    agent = TransformerAgentWrapper(
        action_dim=action_dim,
        n_features=n_features,
        window=window,
        hidden_dim=8,
        num_heads=2,
        num_layers=1,
        update_epochs=1,
        num_minibatches=1,
    )
    model_file = "model.pt"
    agent.save(directory / model_file)
    manifest_path = _write_artifact(
        directory,
        "transformer",
        model_file,
        contract,
        window,
        n_features,
        action_dim,
    )
    return agent, manifest_path


class FixedPolicy:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=np.float64)
        self.action_dim = len(self.probabilities)
        self.obs_dim = 3
        self.reset_count = 0
        self.observed = []

    def reset(self):
        self.reset_count += 1

    def act(self, obs):
        action = int(np.argmax(self.probabilities))
        return PolicyOutput(
            action,
            self.probabilities,
            float(self.probabilities[action]),
        )

    def observe_executed(self, action):
        self.observed.append(int(action))


def test_transformer_tokenization_and_gae():
    window, n_features = 3, 2
    market = np.arange(window * n_features, dtype=np.float32)
    account = np.asarray([1.0, 0.2, 3.0, 0.1, 0.95], dtype=np.float32)
    observation = np.concatenate((market, account))
    tokens = tokenize_observation(observation, window, n_features).numpy()
    assert tokens.shape == (window, n_features + 5)
    np.testing.assert_array_equal(tokens[:, :n_features].reshape(-1), market)
    np.testing.assert_array_equal(tokens[:, n_features:], np.tile(account, (window, 1)))

    advantages, returns = compute_gae(
        [1.0, 1.0, 1.0],
        [0.5, 0.6, 0.7],
        [0.0, 0.0, 1.0],
        last_value=8.0,
        gamma=0.9,
        lam=0.8,
    )
    torch.testing.assert_close(
        advantages,
        torch.tensor([1.93712, 1.246, 0.3]),
        atol=1e-5,
        rtol=0,
    )
    torch.testing.assert_close(
        returns,
        torch.tensor([2.43712, 1.846, 1.0]),
        atol=1e-5,
        rtol=0,
    )


def test_transformer_train_attention_and_checkpoint(tmp_path):
    torch.manual_seed(2)
    agent = TransformerAgentWrapper(
        action_dim=3,
        n_features=2,
        window=3,
        hidden_dim=8,
        num_heads=2,
        num_layers=2,
        update_epochs=1,
        num_minibatches=2,
    )
    rng = np.random.default_rng(2)
    obs = rng.normal(size=(4, agent.obs_dim)).astype(np.float32)
    before = [parameter.detach().clone() for parameter in agent.actor.parameters()]
    metrics = agent.train_step(
        {
            "obs": obs,
            "actions": np.asarray([0, 1, 2, 1]),
            "old_logp": np.full(4, -np.log(3), dtype=np.float32),
            "advantages": np.asarray([1.0, -0.5, 0.3, 0.8], dtype=np.float32),
            "returns": np.asarray([0.1, -0.2, 0.3, 0.4], dtype=np.float32),
        }
    )
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, agent.actor.parameters())
    )
    weights = agent.get_attention_weights(obs[0])
    assert weights.shape == (2, 3, 3)
    np.testing.assert_allclose(weights.sum(axis=-1), 1.0, atol=1e-6)

    checkpoint = tmp_path / "transformer.pt"
    agent.save(checkpoint)
    restored = TransformerAgentWrapper.from_checkpoint(checkpoint)
    np.testing.assert_allclose(
        agent.policy_probs(obs).detach().numpy(),
        restored.policy_probs(obs).detach().numpy(),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize(
    ("probabilities", "expected_action", "expected_consensus"),
    [
        ([[0.1, 0.9], [0.2, 0.8]], 1, True),
        ([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], 0, False),
    ],
)
def test_ensemble_consensus_and_fallback(
    probabilities, expected_action, expected_consensus
):
    members = [FixedPolicy(probs) for probs in probabilities]
    policy = EnsemblePolicy(members, min_agreement=0.6)
    output = policy.act(np.zeros(3, dtype=np.float32))
    assert output.action == expected_action
    assert output.info["consensus"] is expected_consensus
    assert np.isfinite(output.info["uncertainty"])
    assert np.isfinite(output.info["epistemic"])
    assert np.isfinite(output.info["epistemic_uncertainty"])
    if not expected_consensus:
        assert output.info["agreement"] == pytest.approx(0.5)


def test_ensemble_weights_padding_hard_vote_and_lifecycle():
    obs = np.zeros(3, dtype=np.float32)
    first = FixedPolicy([0.1, 0.9, 0.0])
    second = FixedPolicy([0.8, 0.2, 0.0])
    assert EnsemblePolicy(
        [first, second], min_agreement=0.0, weights=[0.1, 0.9]
    ).act(obs).action == 0
    assert EnsemblePolicy(
        [first, second], min_agreement=0.0, weights=[0.9, 0.1]
    ).act(obs).action == 1

    mixed = EnsemblePolicy(
        [FixedPolicy([0.2, 0.8]), FixedPolicy([0.0, 0.0, 1.0])],
        min_agreement=0.5,
    )
    mixed_output = mixed.act(obs)
    np.testing.assert_allclose(mixed_output.probs, [0.1, 0.4, 0.5])
    assert mixed_output.info["member_actions"] == [1, 2]

    soft_members = [
        FixedPolicy([0.51, 0.49]),
        FixedPolicy([0.51, 0.49]),
        FixedPolicy([0.0, 1.0]),
    ]
    soft = EnsemblePolicy(soft_members, min_agreement=0.0, vote="soft")
    hard = EnsemblePolicy(
        [FixedPolicy([0.51, 0.49]), FixedPolicy([0.51, 0.49]), FixedPolicy([0.0, 1.0])],
        min_agreement=0.0,
        vote="hard",
    )
    assert soft.act(obs).action == 1
    assert hard.act(obs).action == 0
    mixed.reset()
    mixed.observe_executed(2)
    assert mixed.members[0].reset_count == 1
    assert mixed.members[0].observed == [0]


def test_ensemble_agent_accepts_array_and_tuple_outputs():
    class ArrayAgent:
        def act(self, obs):
            return np.asarray([0.1, 0.9])

    class TupleAgent:
        def act(self, obs):
            return np.asarray([0.8, 0.2]), None

    assert (
        EnsembleAgent(ArrayAgent, num_models=1).act(
            np.zeros(1), use_consensus=False
        )[0]
        == 1
    )
    assert (
        EnsembleAgent(TupleAgent, num_models=1).act(
            np.zeros(1), use_consensus=False
        )[0]
        == 0
    )


def test_mcts_search_backup_and_three_action_initialization():
    torch.manual_seed(9)
    obs_dim = 4 * 2 + 5
    agent = _tiny_dreamer(obs_dim, action_dim=3, seq_len=2)
    mcts = MCTS(agent, num_simulations=5, seed=17)
    h, z = agent.rssm.initial_state(1, agent.device)
    action, stats = mcts.search(h, z)
    assert 0 <= action < 3
    assert stats["root_visits"] == 5
    assert sum(stats["visit_counts"]) == 5
    assert stats["visit_distribution"].sum() == pytest.approx(1.0)
    assert stats["probs"] == pytest.approx(stats["visit_distribution"])
    repeated_action, repeated_stats = mcts.search(h, z)
    assert repeated_action == action
    assert repeated_stats["visit_counts"] == stats["visit_counts"]
    assert repeated_stats["q_values"] == pytest.approx(stats["q_values"])

    root = MCTSNode(torch.zeros(1, 8), torch.zeros(1, 2))
    child = MCTSNode(
        torch.zeros(1, 8),
        torch.zeros(1, 2),
        parent=root,
        action=1,
        reward=0.25,
    )
    root.children[1] = child
    backup_search = MCTS(agent, num_simulations=1, gamma=0.5)
    backup_search.backup([root, child], leaf_value=2.0)
    assert root.visit_count == child.visit_count == 1
    assert child.value_sum == pytest.approx(1.25)
    assert root.value_sum == pytest.approx(1.25)

    observed_actions = []
    dreamer = _tiny_dreamer(obs_dim, action_dim=3, seq_len=2)
    observe = dreamer.rssm.observe

    def capture_observe(embed, action_tensor, h_state, z_state):
        observed_actions.append(action_tensor.detach().cpu().numpy().copy())
        return observe(embed, action_tensor, h_state, z_state)

    dreamer.rssm.observe = capture_observe
    wrapper = DreamerMCTSAgent(dreamer, num_simulations=2)
    wrapper.act(np.zeros(obs_dim, dtype=np.float32))
    assert observed_actions[0].shape == (1, 3)
    np.testing.assert_array_equal(observed_actions[0], np.zeros((1, 3)))


def _task_segment(seed, length=40, window=2):
    rng = np.random.default_rng(seed)
    return {
        "features": rng.normal(size=(length, 2)).astype(np.float32),
        "returns": rng.normal(0.0, 0.001, size=length).astype(np.float32),
        "timestamps": pd.date_range("2020-01-01", periods=length, freq="h").to_numpy(),
        "window": window,
    }


def test_maml_regime_labels_are_causal_and_return_segments():
    returns = np.full(1800, 0.001, dtype=np.float32)
    features = np.ones((len(returns), 2), dtype=np.float32)
    timestamps = pd.date_range("2020-01-01", periods=len(returns), freq="h").to_numpy()
    labels = MarketRegimeGenerator._labels(returns)
    extended = MarketRegimeGenerator._labels(
        np.concatenate((returns, np.full(200, -0.02, dtype=np.float32)))
    )
    np.testing.assert_array_equal(labels, extended[: len(labels)])

    regimes = MarketRegimeGenerator.generate_regimes(
        features,
        returns,
        timestamps,
        window=4,
        min_len=512,
    )
    assert regimes
    assert regimes[0]["label"] in {
        "trend_up",
        "trend_down",
        "range",
        "high_vol",
        "low_vol",
    }
    assert regimes[0]["start"] < regimes[0]["end"]
    assert len(regimes[0]["support"]["features"]) > len(regimes[0]["query"]["features"])


def test_maml_builds_replay_and_meta_train_updates_copy_only():
    torch.manual_seed(15)
    agent = _tiny_dreamer(obs_dim=9, action_dim=3, seq_len=2)
    support = _task_segment(4, length=40, window=2)
    query = _task_segment(5, length=40, window=2)
    buffer = build_task_buffer(
        agent,
        support,
        steps=len(support["returns"]) - support["window"] - 1,
        seed=2,
    )
    assert len(buffer) == 37
    assert buffer.sample(1)["obs"].shape == (1, 2, 9)

    params_before = [parameter.detach().clone() for parameter in agent.world_model_params]
    trader = MAMLTrader(agent, adapt_lr=1e-4, adapt_steps=1)
    result = trader.meta_train(
        [{"support": support, "query": query}],
        num_epochs=1,
        tasks_per_batch=1,
        batch_size=1,
    )
    assert np.isfinite(result[0]["meta_loss"])
    assert np.isfinite(result[0]["support_loss"])
    assert any(
        not torch.equal(old, new)
        for old, new in zip(params_before, agent.world_model_params)
    )

    another_buffer = build_task_buffer(
        agent,
        query,
        steps=len(query["returns"]) - query["window"] - 1,
        seed=3,
    )
    params_before_adapt = [
        parameter.detach().clone() for parameter in agent.world_model_params
    ]
    adapted = trader.fast_adapt(another_buffer, steps=1, batch_size=1)
    assert adapted is not agent
    assert all(
        torch.equal(old, current)
        for old, current in zip(params_before_adapt, agent.world_model_params)
    )


def test_market_perturbations_are_noop_one_step_and_stop_hunt():
    actions = [
        np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
    ]
    base = _market_env(seed=24)
    noop = _market_env(seed=24)
    obs_base = base.reset()
    obs_noop = noop.reset()
    np.testing.assert_array_equal(obs_base, obs_noop)
    for action in actions:
        noop.set_perturbation()
        obs_base, reward_base, done_base, info_base = base.step(action)
        obs_noop, reward_noop, done_noop, info_noop = noop.step(action)
        np.testing.assert_array_equal(obs_base, obs_noop)
        assert reward_base == reward_noop
        assert done_base == done_noop
        assert info_base == info_noop

    returns = np.full(30, 0.0001, dtype=np.float32)
    normal = _market_env(length=30, seed=8, returns=returns, spread=0.001, slippage=0.0)
    perturbed = _market_env(length=30, seed=8, returns=returns, spread=0.001, slippage=0.0)
    normal.reset()
    perturbed.reset()
    perturbed.set_perturbation(spread_mult=3.0)
    _, _, _, normal_entry = normal.step(actions[1])
    _, _, _, perturbed_entry = perturbed.step(actions[1])
    assert perturbed_entry["cost"] > normal_entry["cost"]
    normal_equity_before_exit = normal.equity
    perturbed_equity_before_exit = perturbed.equity
    _, _, _, normal_exit = normal.step(actions[2])
    _, _, _, perturbed_exit = perturbed.step(actions[2])
    assert (
        normal_exit["cost"] * normal_equity_before_exit
        == pytest.approx(perturbed_exit["cost"] * perturbed_equity_before_exit)
    )

    stop_returns = np.zeros(30, dtype=np.float32)
    stop_returns[4] = -0.006
    normal_stop = _market_env(
        length=30,
        seed=5,
        returns=stop_returns,
        spread=0,
        commission=0,
        slippage=0,
        stop_loss=0.01,
    )
    hunted_stop = _market_env(
        length=30,
        seed=5,
        returns=stop_returns,
        spread=0,
        commission=0,
        slippage=0,
        stop_loss=0.01,
    )
    normal_stop.reset()
    hunted_stop.reset()
    hunted_stop.set_perturbation(adverse_gap=0.005)
    _, _, _, normal_info = normal_stop.step(actions[0])
    _, _, _, hunted_info = hunted_stop.step(actions[0])
    assert normal_info["forced_close"] is False
    assert hunted_info["forced_close"] is True


def test_market_maker_rate_cap_and_self_play_trains_dreamer():
    market_maker = MarketMakerAgent(
        state_dim=9,
        hidden_dim=8,
        max_manipulation_rate=0.2,
    )
    with torch.no_grad():
        for parameter in market_maker.policy.parameters():
            parameter.zero_()
        market_maker.policy[-1].bias[:] = torch.tensor([-20.0, 20.0, -20.0, -20.0])
    for _ in range(100):
        market_maker.select_action(1, np.zeros(9, dtype=np.float32))
    assert market_maker.manipulation_count <= 20
    assert market_maker.manipulation_rate <= 0.2

    features = np.zeros((60, 1), dtype=np.float32)
    returns = np.random.default_rng(6).normal(0.0, 0.001, size=60).astype(np.float32)
    env = _market_env(
        length=60,
        window=4,
        seed=6,
        features=features,
        returns=returns,
    )
    trader = _tiny_dreamer(obs_dim=9, action_dim=3, seq_len=2)
    maker = MarketMakerAgent(state_dim=env.observation_space, hidden_dim=8)
    trainer = SelfPlayTrainer(
        trader,
        maker,
        env,
        train_every=1,
        batch_size=1,
    )
    history = trainer.train(num_epochs=1, steps_per_epoch=4)
    assert history["trader_updates"] >= 1
    assert trader.training_step >= 1


def test_registry_loads_transformer_ensemble_and_rejects_hash_mismatch(tmp_path):
    frame = sample_df(n=620)
    data_path = tmp_path / "bars.csv"
    frame.to_csv(data_path, index=False)
    train_end = frame["time"].iloc[420].isoformat()
    prepared = prepare_data(data_path, train_end=train_end, window=4)
    contract = prepared.contract
    first_dir = tmp_path / "members" / "first"
    second_dir = tmp_path / "members" / "second"
    _, first_manifest = _make_transformer_artifact(
        first_dir, contract, window=4, n_features=len(prepared.feature_names)
    )
    _, second_manifest = _make_transformer_artifact(
        second_dir, contract, window=4, n_features=len(prepared.feature_names)
    )

    policy, manifest = load_policy(first_manifest)
    output = policy.act(np.zeros(manifest.obs_dim, dtype=np.float32))
    assert isinstance(output, PolicyOutput)
    assert output.probs.sum() == pytest.approx(1.0)
    assert output.action < manifest.action_dim

    ensemble_dir = tmp_path / "ensembles"
    ensemble_args = SimpleNamespace(
        from_manifests=[str(first_manifest), str(second_manifest)],
        base_type="transformer",
        members=2,
        weights=[0.7, 0.3],
        min_agreement=0.5,
        vote="soft",
        run_name="assembled",
        artifact_root=str(ensemble_dir),
        data=str(data_path),
        eval_start=train_end,
        train_end=train_end,
        test_end=None,
        macro=None,
    )
    assembled = train_ensemble(ensemble_args)
    ensemble_policy, ensemble_manifest = load_policy(assembled / "manifest.json")
    assert len(ensemble_policy.members) == 2
    assert all(not Path(member).is_absolute() for member in ensemble_manifest.members)

    mismatch_dir = tmp_path / "members" / "mismatch"
    mismatch_contract = {**contract, "hash": "different-contract"}
    _, mismatch_manifest = _make_transformer_artifact(
        mismatch_dir,
        mismatch_contract,
        window=4,
        n_features=len(prepared.feature_names),
    )
    bad_dir = tmp_path / "bad-ensemble"
    (bad_dir / "ensemble.json").parent.mkdir(parents=True, exist_ok=True)
    (bad_dir / "ensemble.json").write_text("{}", encoding="utf-8")
    bad_manifest = _write_artifact(
        bad_dir,
        "ensemble",
        "ensemble.json",
        contract,
        window=4,
        n_features=len(prepared.feature_names),
        action_dim=3,
        members=[
            "../members/first/manifest.json",
            f"../{mismatch_manifest.relative_to(tmp_path)}",
        ],
        extra={"min_agreement": 0.5, "vote": "soft"},
    )
    with pytest.raises(ModelArtifactError, match="same contract hash"):
        load_policy(bad_manifest)


def test_registry_loads_dreamer_mcts_manifest(tmp_path):
    window, n_features = 3, 2
    contract = {"hash": "dreamer-contract"}
    dreamer_dir = tmp_path / "dreamer"
    dreamer_dir.mkdir()
    agent = _tiny_dreamer(window * n_features + 5, action_dim=3, seq_len=2)
    agent.save(dreamer_dir / "model.pt")
    dreamer_manifest = _write_artifact(
        dreamer_dir,
        "dreamer",
        "model.pt",
        contract,
        window,
        n_features,
        3,
    )
    mcts_dir = make_manifest(
        SimpleNamespace(
            dreamer=str(dreamer_manifest),
            simulations=2,
            c_puct=1.0,
            artifact_root=str(tmp_path / "artifacts"),
            run_name="mcts",
        )
    )
    policy, manifest = load_policy(mcts_dir / "manifest.json")
    assert isinstance(policy, DreamerMCTSPolicy)
    assert policy.action_dim == 3
    output = policy.act(np.zeros(manifest.obs_dim, dtype=np.float32))
    assert output.action in range(3)


def test_transformer_trainer_writes_complete_artifact(tmp_path):
    frame = sample_df(n=620)
    data_path = tmp_path / "bars.csv"
    frame.to_csv(data_path, index=False)
    out_dir = train_transformer(
        SimpleNamespace(
            data=str(data_path),
            macro=None,
            train_end=frame["time"].iloc[420].isoformat(),
            test_end=None,
            steps=4,
            rollout_steps=2,
            batch_size=2,
            update_epochs=1,
            num_minibatches=1,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            window=8,
            hidden_dim=8,
            num_heads=2,
            num_layers=1,
            seed=14,
            run_name="tiny-transformer",
            artifact_root=str(tmp_path / "artifacts"),
            allow_short=True,
        )
    )
    assert (out_dir / "model.pt").is_file()
    assert (out_dir / "feature_contract.json").is_file()
    assert (out_dir / "manifest.json").is_file()
    assert (out_dir / "evaluation.json").is_file()
