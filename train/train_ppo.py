import argparse
import sys
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.model_artifacts import artifact_dir
from env.gym_env import GymTradingEnv
from models.policy import PpoPolicy
from train.common import write_model_artifact
from train.data import prepare_data
from train.evaluate import evaluate_policy, write_evaluation


def _arg(args, name, default):
    return getattr(args, name, default)


def train(args) -> Path:
    data = prepare_data(
        _arg(args, "data", "data/xauusd_1h.csv"),
        train_end=_arg(args, "train_end", "2022-01-01"),
        window=int(_arg(args, "window", 64)),
        macro_csv=_arg(args, "macro", None),
        test_end=_arg(args, "test_end", None),
    )
    allow_short = bool(_arg(args, "allow_short", True))
    n_envs = int(_arg(args, "n_envs", 1))
    seed = int(_arg(args, "seed", 42))
    max_episode_steps = _arg(args, "max_episode_steps", 4096)
    env_fns = [
        lambda i=i: GymTradingEnv(
            data.X_train,
            data.r_train,
            timestamps=data.ts_train,
            window=data.window,
            allow_short=allow_short,
            max_episode_steps=max_episode_steps,
            seed=seed + i,
        )
        for i in range(n_envs)
    ]
    vec_env = (
        SubprocVecEnv(env_fns)
        if bool(_arg(args, "subproc", False))
        else DummyVecEnv(env_fns)
    )
    resume = _arg(args, "resume", None)
    try:
        if resume:
            model = PPO.load(resume, env=vec_env, device="cpu")
        else:
            net_arch = _arg(args, "net_arch", _arg(args, "hidden_sizes", [256, 256]))
            model = PPO(
                "MlpPolicy",
                vec_env,
                policy_kwargs={"net_arch": list(net_arch)},
                n_steps=int(_arg(args, "n_steps", 1024)),
                batch_size=int(_arg(args, "batch_size", 256)),
                gamma=float(_arg(args, "gamma", 0.99)),
                learning_rate=float(_arg(args, "learning_rate", 3e-4)),
                seed=seed,
                device="cpu",
                verbose=int(_arg(args, "verbose", 0)),
            )

        timesteps = int(_arg(args, "timesteps", _arg(args, "steps", 1_000_000)))
        if timesteps < 1:
            raise ValueError("timesteps must be positive")
        chunk_steps = max(1, int(_arg(args, "chunk_steps", 50_000)))
        out_dir = artifact_dir(
            Path(_arg(args, "artifact_root", "artifacts/models")),
            "ppo",
            _arg(args, "run_name", None),
        )
        remaining = timesteps
        while remaining > 0:
            current_chunk = min(chunk_steps, remaining)
            target = model.num_timesteps + current_chunk
            model.learn(total_timesteps=target, reset_num_timesteps=False)
            model.save(out_dir / f"checkpoint_{model.num_timesteps}.zip")
            remaining -= current_chunk

        model_path = out_dir / "model.zip"
        model.save(model_path)
    finally:
        vec_env.close()

    action_dim = int(model.action_space.n)
    write_model_artifact(
        out_dir,
        model_type="ppo",
        model_file=model_path.name,
        window=data.window,
        n_features=len(data.feature_names),
        obs_dim=int(model.observation_space.shape[0]),
        action_dim=action_dim,
        allow_short=allow_short,
        contract=data.contract,
        symbol=_arg(args, "symbol", "XAUUSD"),
        timeframe=_arg(args, "timeframe", "H1"),
        train_start=data.ts_train[0],
        train_end=_arg(args, "train_end", "2022-01-01"),
        test_end=_arg(args, "test_end", None),
        hyperparams={
            "timesteps": timesteps,
            "n_envs": n_envs,
            "n_steps": int(_arg(args, "n_steps", 1024)),
            "batch_size": int(_arg(args, "batch_size", 256)),
            "net_arch": list(_arg(args, "net_arch", _arg(args, "hidden_sizes", [256, 256]))),
        },
    )
    if len(data.X_test) <= data.window + 2:
        raise ValueError("test period must contain more than window+2 feature rows")
    policy = PpoPolicy(model_path)
    metrics = evaluate_policy(
        policy,
        data.X_test,
        data.r_test,
        data.ts_test,
        {"window": data.window, "allow_short": allow_short},
    )
    write_evaluation(
        out_dir,
        metrics,
        data.ts_test[0],
        data.ts_test[-1],
        data.contract["hash"],
    )
    return out_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train PPO on the shared trading environment")
    parser.add_argument("--data", default="data/xauusd_1h.csv")
    parser.add_argument("--macro")
    parser.add_argument("--train-end", dest="train_end", default="2022-01-01")
    parser.add_argument("--test-end", dest="test_end")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--subproc", action="store_true")
    parser.add_argument("--run-name")
    parser.add_argument("--artifact-root", default="artifacts/models")
    parser.add_argument("--resume")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--chunk-steps", type=int, default=50_000)
    parser.add_argument("--max-episode-steps", type=int, default=4096)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--allow-short", dest="allow_short", action="store_true")
    group.add_argument("--long-only", dest="allow_short", action="store_false")
    parser.set_defaults(allow_short=True)
    args = parser.parse_args(argv)
    print(train(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
