import argparse
import json
import math
import os
import sys
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
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


class DashboardCallback(BaseCallback):
    """Streams training step metrics to DASHBOARD_METRICS_PATH for dashboard plotting."""

    def __init__(self, metrics_path: str, save_freq: int = 1024, verbose: int = 0):
        super().__init__(verbose)
        self.metrics_path = Path(metrics_path)
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_freq = max(1, save_freq)
        self._last_save_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save_step >= self.save_freq:
            self._last_save_step = self.num_timesteps
            name_to_value = getattr(self.logger, "name_to_value", {})
            sample = {"training_step": int(self.num_timesteps)}
            # Map SB3 logger keys to standard loss plot keys
            key_map = {
                "train/value_loss": "value_loss",
                "train/policy_gradient_loss": "policy_loss",
                "train/entropy_loss": "entropy",
                "train/approx_kl": "kl_loss",
                "train/loss": "world_model_loss",
                "rollout/ep_rew_mean": "reward_loss",
            }
            has_metric = False
            for sb3_key, plot_key in key_map.items():
                if sb3_key in name_to_value:
                    val = name_to_value[sb3_key]
                    if val is not None and math.isfinite(float(val)):
                        sample[plot_key] = float(val)
                        has_metric = True
            if has_metric:
                try:
                    with self.metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(sample, allow_nan=False) + "\n")
                except OSError:
                    pass
        return True


def _arg(args, name, default):
    return getattr(args, name, default)


def train(args) -> Path:
    resume = _arg(args, "resume", None)
    resume_file = None
    if resume:
        resume_path = Path(resume)
        if resume_path.is_dir():
            candidate = resume_path / "model.zip"
            if candidate.exists():
                resume_file = candidate
            else:
                ckpts = sorted(resume_path.glob("checkpoint_*.zip"), key=lambda p: p.stat().st_mtime)
                if ckpts:
                    resume_file = ckpts[-1]
                else:
                    raise FileNotFoundError(f"No model.zip or checkpoint_*.zip found in {resume_path}")
            manifest_file = resume_path / "manifest.json"
        else:
            resume_file = resume_path
            manifest_file = resume_path.parent / "manifest.json"

        # Auto-detect parameters from manifest if not explicitly given on CLI
        if manifest_file.exists():
            try:
                manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
                if getattr(args, "window", None) is None and "window" in manifest_data:
                    args.window = int(manifest_data["window"])
                if getattr(args, "train_end", None) is None and "train_end" in manifest_data:
                    args.train_end = manifest_data["train_end"]
                if getattr(args, "test_end", None) is None and "test_end" in manifest_data and manifest_data["test_end"]:
                    args.test_end = manifest_data["test_end"]
                manifest_allow_short = bool(manifest_data.get("allow_short", True))
                if getattr(args, "allow_short", None) is None:
                    args.allow_short = manifest_allow_short
                elif args.allow_short != manifest_allow_short:
                    print(f"⚠️ Notice: Checkpoint network was trained with allow_short={manifest_allow_short} (Discrete({3 if manifest_allow_short else 2})).")
                    if not args.allow_short and manifest_allow_short:
                        print("   Cannot change action space to Discrete(2) on resume without crashing PyTorch.")
                        print("   Auto-retaining allow_short=True and activating --short-penalty 0.001 to penalize & suppress short trades.")
                        args.allow_short = True
                        if getattr(args, "short_penalty", 0.0) == 0.0:
                            args.short_penalty = 0.001
                    else:
                        args.allow_short = manifest_allow_short
                manifest_n_features = int(manifest_data.get("n_features", 15))
                if manifest_n_features <= 15 and getattr(args, "macro", None):
                    print(f"✓ Notice: Checkpoint was trained without macro features (n_features={manifest_n_features}). Disabling --macro for weights compatibility.")
                    args.macro = None
                elif manifest_n_features > 15 and not getattr(args, "macro", None):
                    print(f"✓ Notice: Checkpoint was trained with macro features (n_features={manifest_n_features}). Auto-enabling --macro.")
                    args.macro = "data/macro_daily.csv"
                print(f"✓ Resuming with manifest defaults: window={args.window}, train_end={args.train_end}, allow_short={args.allow_short}, macro={bool(args.macro)}")
            except Exception as e:
                print(f"Warning: Could not read manifest {manifest_file}: {e}")

    window = int(_arg(args, "window", 64) or 64)
    train_end = _arg(args, "train_end", "2022-01-01") or "2022-01-01"

    data = prepare_data(
        _arg(args, "data", "data/xauusd_h1.csv"),
        train_end=train_end,
        window=window,
        macro_csv=_arg(args, "macro", None),
        test_end=_arg(args, "test_end", None),
    )
    allow_short = bool(_arg(args, "allow_short", True))
    n_envs = int(_arg(args, "n_envs", 1))
    seed = int(_arg(args, "seed", 42))
    max_episode_steps = _arg(args, "max_episode_steps", 4096)

    # Qadam 1: Penalties and incentives to stop hyperactive fee bleed
    turnover_penalty = float(_arg(args, "turnover_penalty", 0.0005))
    hold_bonus = float(_arg(args, "hold_bonus", 0.00005))
    min_hold_bars = int(_arg(args, "min_hold_bars", 4))
    early_exit_penalty = float(_arg(args, "early_exit_penalty", 0.0005))
    drawdown_penalty = float(_arg(args, "drawdown_penalty", 0.5))
    short_penalty = float(_arg(args, "short_penalty", 0.0))

    print("\n" + "=" * 65)
    print("📋 Environment Configuration (Qadam 1 Anti-Churn Guards):")
    print(f"   Turnover penalty:   {turnover_penalty * 1e4:.1f} bps per flip")
    print(f"   Hold bonus:         {hold_bonus * 1e4:.1f} bps per bar")
    print(f"   Min hold bars:      {min_hold_bars} bars (~{min_hold_bars}h)")
    print(f"   Early exit penalty: {early_exit_penalty * 1e4:.1f} bps")
    print(f"   Drawdown penalty:   {drawdown_penalty:.2f}")
    if short_penalty > 0:
        print(f"   Short penalty:      {short_penalty * 1e4:.1f} bps per bar (suppress shorts)")
    print("=" * 65 + "\n")

    env_fns = [
        lambda i=i: GymTradingEnv(
            data.X_train,
            data.r_train,
            timestamps=data.ts_train,
            window=data.window,
            allow_short=allow_short,
            max_episode_steps=max_episode_steps,
            seed=seed + i,
            turnover_penalty=turnover_penalty,
            hold_bonus=hold_bonus,
            min_hold_bars=min_hold_bars,
            early_exit_penalty=early_exit_penalty,
            drawdown_penalty=drawdown_penalty,
            short_penalty=short_penalty,
        )
        for i in range(n_envs)
    ]
    vec_env = (
        SubprocVecEnv(env_fns)
        if bool(_arg(args, "subproc", False))
        else DummyVecEnv(env_fns)
    )

    try:
        verbose = int(_arg(args, "verbose", 1))
        if resume_file:
            print(f"🔄 Loading existing weights from: {resume_file}")
            model = PPO.load(str(resume_file), env=vec_env, device="cpu")
            model.verbose = verbose
            if _arg(args, "learning_rate", None) is not None:
                lr = float(args.learning_rate)
                model.learning_rate = lr
                model.lr_schedule = lambda _: lr
                for param_group in model.policy.optimizer.param_groups:
                    param_group["lr"] = lr
                print(f"✓ Optimizer learning rate updated to: {lr}")
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
                verbose=verbose,
            )

        timesteps = int(_arg(args, "timesteps", _arg(args, "steps", 1_000_000)))
        if timesteps < 1:
            raise ValueError("timesteps must be positive")
        chunk_steps = max(1, int(_arg(args, "chunk_steps", 50_000)))

        run_name = _arg(args, "run_name", None)
        if resume_file:
            if run_name:
                out_dir = Path(_arg(args, "artifact_root", "artifacts/models")) / run_name
            else:
                out_dir = resume_file.parent if resume_file.is_file() else resume_path
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = artifact_dir(
                Path(_arg(args, "artifact_root", "artifacts/models")),
                "ppo",
                run_name,
            )
        metrics_env_path = os.environ.get("DASHBOARD_METRICS_PATH")
        callback = DashboardCallback(metrics_env_path, save_freq=int(_arg(args, "n_steps", 1024))) if metrics_env_path else None
        target_timesteps = model.num_timesteps + timesteps

        start_timesteps = model.num_timesteps
        print("\n" + "=" * 65)
        print(f"🚀 Training: {start_timesteps:,} -> {target_timesteps:,} steps")
        print(f"   Adding: {timesteps:,} steps | Chunk size: {chunk_steps:,}")
        print(f"   Artifact Output: {out_dir}")
        print("=" * 65 + "\n")
        sys.stdout.flush()

        while model.num_timesteps < target_timesteps:
            current_chunk = min(chunk_steps, target_timesteps - model.num_timesteps)
            model.learn(total_timesteps=current_chunk, reset_num_timesteps=False, callback=callback)
            ckpt_name = f"checkpoint_{model.num_timesteps}.zip"
            model.save(out_dir / ckpt_name)
            try:
                write_model_artifact(
                    out_dir,
                    model_type="ppo",
                    model_file=ckpt_name,
                    window=data.window,
                    n_features=len(data.feature_names),
                    obs_dim=int(model.observation_space.shape[0]),
                    action_dim=int(model.action_space.n),
                    allow_short=allow_short,
                    contract=data.contract,
                    symbol=_arg(args, "symbol", "XAUUSD"),
                    timeframe=_arg(args, "timeframe", "H1"),
                    train_start=data.ts_train[0],
                    train_end=train_end,
                    test_end=_arg(args, "test_end", None),
                    hyperparams={
                        "timesteps": int(model.num_timesteps),
                        "n_envs": n_envs,
                        "n_steps": int(_arg(args, "n_steps", 1024)),
                        "batch_size": int(_arg(args, "batch_size", 256)),
                        "net_arch": list(net_arch),
                    },
                )
            except Exception:
                pass
            added_done = model.num_timesteps - start_timesteps
            pct = (added_done / timesteps) * 100
            print(f"[{model.num_timesteps:,} / {target_timesteps:,} steps] ({pct:.1f}%) | Checkpoint: checkpoint_{model.num_timesteps}.zip")
            sys.stdout.flush()

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
        train_end=train_end,
        test_end=_arg(args, "test_end", None),
        hyperparams={
            "timesteps": int(model.num_timesteps),
            "added_timesteps": timesteps,
            "n_envs": n_envs,
            "n_steps": int(_arg(args, "n_steps", 1024)),
            "batch_size": int(_arg(args, "batch_size", 256)),
            "net_arch": list(_arg(args, "net_arch", _arg(args, "hidden_sizes", [256, 256]))),
            "turnover_penalty": turnover_penalty,
            "hold_bonus": hold_bonus,
            "min_hold_bars": min_hold_bars,
            "early_exit_penalty": early_exit_penalty,
            "drawdown_penalty": drawdown_penalty,
            "short_penalty": short_penalty,
        },
    )
    if len(data.X_test) > data.window + 2:
        print("\nRunning post-training evaluation...")
        policy = PpoPolicy(model_path)
        metrics = evaluate_policy(
            policy,
            data.X_test,
            data.r_test,
            data.ts_test,
            {"window": data.window, "allow_short": allow_short},
        )
        eval_path = write_evaluation(
            out_dir,
            metrics,
            data.ts_test[0],
            data.ts_test[-1],
            data.contract["hash"],
        )
        print(f"✓ Evaluation saved to: {eval_path}")
        print(f"  Passed Promotion Gate: {metrics.get('sharpe', 0) >= 0.5 and metrics.get('max_dd_pct', 100) <= 20 and metrics.get('total_return_pct', -1) > 0}")
    else:
        print(f"Test period ({len(data.X_test)} bars) <= window+2 ({data.window + 2}); skipping post-train evaluation.")
    return out_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train PPO on the shared trading environment")
    parser.add_argument("--data", default="data/xauusd_h1.csv")
    parser.add_argument("--macro")
    parser.add_argument("--train-end", dest="train_end", default=None)
    parser.add_argument("--test-end", dest="test_end")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--steps", dest="timesteps", type=int, help="Alias for --timesteps")
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-envs", type=int, default=2)
    parser.add_argument("--subproc", action="store_true")
    parser.add_argument("--run-name")
    parser.add_argument("--artifact-root", default="artifacts/models")
    parser.add_argument("--resume")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--chunk-steps", type=int, default=50_000)
    parser.add_argument("--max-episode-steps", type=int, default=4096)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--turnover-penalty", type=float, default=0.0005,
                        help="Turnover penalty per position change delta (default: 0.0005 = 5 bps)")
    parser.add_argument("--hold-bonus", type=float, default=0.00005,
                        help="Holding bonus per bar for staying in position (default: 0.00005 = 0.5 bps)")
    parser.add_argument("--min-hold-bars", type=int, default=4,
                        help="Minimum bars to hold trade before voluntary close (default: 4 bars)")
    parser.add_argument("--early-exit-penalty", type=float, default=0.0005,
                        help="Penalty for voluntary exit before min_hold_bars (default: 0.0005)")
    parser.add_argument("--drawdown-penalty", type=float, default=0.5,
                        help="Penalty factor for incremental drawdown (default: 0.5)")
    parser.add_argument("--short-penalty", type=float, default=0.0,
                        help="Penalty per bar for holding short positions (default: 0.0)")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--allow-short", dest="allow_short", action="store_true")
    group.add_argument("--long-only", dest="allow_short", action="store_false")
    parser.set_defaults(allow_short=None)
    args = parser.parse_args(argv)
    print(train(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
