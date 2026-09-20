"""
Train DreamerV3 Agent on XAUUSD Trading

This is the "Baby Stockfish" - Phase 1 of PROJECT GOD MODE
"""

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.make_features import make_features
from models.dreamer_agent import DreamerV3Agent
from env.dreamer_trading_env import DEFAULT_ENV_KWARGS, EVAL_ENV_OVERRIDES, RealisticTradingEnv


WINDOW = 64
TRAIN_END_DATE = "2022-01-01"
ENV_KWARGS = dict(DEFAULT_ENV_KWARGS, window=WINDOW)

# DreamerV3 hyperparameters
BATCH_SIZE = 16
PREFILL_STEPS = 5_000  # Random exploration to fill buffer
TRAIN_STEPS = 100_000  # Training steps
TRAIN_EVERY = 4  # Train every N environment steps
SAVE_EVERY = 10_000

SAVE_DIR = "train/dreamer"
SAVE_PREFIX = "dreamer_xauusd"


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train DreamerV3 on XAUUSD')
    parser.add_argument('--steps', type=int, default=TRAIN_STEPS,
                        help='Number of training steps (default: 100000)')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Batch size (default: 16, recommended: 64 for GPU)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device to use (default: auto-detect)')
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)

    # Set device
    if args.device == 'auto':
        # Auto-detect best available (CUDA > MPS > CPU)
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device

    print(f"🚀 Using device: {device}")
    if device == 'mps':
        print("   ⚡ Apple Metal GPU acceleration enabled!")
    elif device == 'cuda':
        print("   ⚡ NVIDIA CUDA GPU acceleration enabled!")

    # Use local variables instead of modifying globals
    batch_size = args.batch_size
    train_steps = args.steps

    print(f"📊 Training config:")
    print(f"   Steps: {train_steps:,}")
    print(f"   Batch size: {batch_size}")

    # Load data
    print("Loading data...")
    # Try to use macro data first
    if os.path.exists("data/xauusd_1h_macro.csv"):
        print("Using MACRO data (DXY, SPX, US10Y) 🚀")
        df, X, r = make_features("data/xauusd_1h_macro.csv", window=WINDOW)
    else:
        print("Using basic XAUUSD data (no macro)")
        df, X, r = make_features("data/xauusd_1h.csv", window=WINDOW)

    # Split train/test
    train_end = np.searchsorted(df["time"].to_numpy(), np.datetime64(TRAIN_END_DATE))
    ts = df["time"].to_numpy()
    X_train, r_train, ts_train = X[:train_end], r[:train_end], ts[:train_end]
    X_test, r_test, ts_test = X[train_end:], r[train_end:], ts[train_end:]

    print(f"Train: {len(X_train)} bars | Test: {len(X_test)} bars")

    # Create environment
    env = RealisticTradingEnv(X_train, r_train, timestamps=ts_train, **ENV_KWARGS)

    # Observation dimension
    obs_dim = env._get_obs().shape[0]
    print(f"Observation dimension: {obs_dim}")

    # Create agent
    print("\nInitializing DreamerV3 Agent...")
    agent = DreamerV3Agent(
        obs_dim=obs_dim,
        action_dim=env.action_space,  # flat, long, short
        device=device,
        embed_dim=256,
        hidden_dim=512,
        stoch_dim=32,
        num_categories=32,
        lr_world_model=3e-4,
        lr_actor=1e-4,
        lr_critic=3e-4,
        gamma=0.99,
        lambda_=0.95,
        horizon=15,
    )

    print("\n" + "="*60)
    print("PHASE 1: Prefill Replay Buffer (Random Exploration)")
    print("="*60)

    obs = env.reset()
    h, z = None, None

    for step in tqdm(range(PREFILL_STEPS), desc="Prefilling"):
        # Random action
        action_onehot = np.zeros(env.action_space, dtype=np.float32)
        action_onehot[np.random.randint(0, env.action_space)] = 1.0

        # Step
        next_obs, reward, done, info = env.step(action_onehot)

        # Store in replay buffer
        agent.replay_buffer.add(obs, action_onehot, reward, done)

        # Next
        obs = next_obs
        if done:
            obs = env.reset()
            h, z = None, None

    print(f"✅ Replay buffer filled with {len(agent.replay_buffer)} transitions")

    print("\n" + "="*60)
    print("PHASE 2: Train DreamerV3 World Model + Policy")
    print("="*60)

    obs = env.reset()
    h, z = None, None
    episode_reward = 0
    episode_count = 0
    step_count = 0

    for train_step in tqdm(range(train_steps), desc="Training"):
        # Act in environment
        action_onehot, (h, z) = agent.act(obs, h, z, deterministic=False)

        # Step
        next_obs, reward, done, info = env.step(action_onehot)

        # Store in replay buffer
        agent.replay_buffer.add(obs, action_onehot, reward, done)

        episode_reward += reward
        step_count += 1

        # Next
        obs = next_obs

        if done:
            print(f"\n  Episode {episode_count}: Reward={episode_reward:.4f}, Equity={info['equity']:.4f}, Steps={step_count}")
            episode_count += 1
            episode_reward = 0
            step_count = 0
            obs = env.reset()
            h, z = None, None

        # Train
        if train_step % TRAIN_EVERY == 0:
            losses = agent.train_step(batch_size=batch_size)

            if losses and train_step % 1000 == 0:
                print(f"\n  Step {train_step}:")
                print(f"    World Model Loss: {losses['world_model_loss']:.4f}")
                print(f"    - Recon: {losses['recon_loss']:.4f}")
                print(f"    - Reward: {losses['reward_loss']:.4f}")
                print(f"    - KL: {losses['kl_loss']:.4f}")
                print(f"    Value Loss: {losses['value_loss']:.4f}")
                print(f"    Policy Loss: {losses['policy_loss']:.4f}")

        # Save checkpoint
        if (train_step + 1) % SAVE_EVERY == 0:
            ckpt_path = f"{SAVE_DIR}/{SAVE_PREFIX}_{(train_step+1)//1000}k.pt"
            agent.save(ckpt_path)
            print(f"\n✅ Saved checkpoint: {ckpt_path}")

    # Final save
    final_path = f"{SAVE_DIR}/{SAVE_PREFIX}_final.pt"
    agent.save(final_path)
    print(f"\n✅ Saved final model: {final_path}")

    print("\n" + "="*60)
    print("PHASE 3: Evaluation on Test Set")
    print("="*60)

    test_env = RealisticTradingEnv(
        X_test, r_test, timestamps=ts_test,
        **{**ENV_KWARGS, **EVAL_ENV_OVERRIDES},
    )
    obs = test_env.reset()
    h, z = None, None

    equities = []
    positions = []

    while True:
        action_onehot, (h, z) = agent.act(obs, h, z, deterministic=True)
        obs, reward, done, info = test_env.step(action_onehot)

        equities.append(info["equity"])
        positions.append(info["position"])

        if done:
            break

    st = test_env.episode_stats()
    positions = np.array(positions)
    final_equity = float(equities[-1])

    print(f"\n📊 Test Results (after spread/commission/slippage/swap):")
    print(f"   Final Equity: {final_equity:.4f}")
    print(f"   Return: {st['return_pct']:.2f}%")
    print(f"   Max Drawdown: {st['max_drawdown_pct']:.2f}%")
    print(f"   Trades: {st['trades']} | Win rate: {st['win_rate']:.1%} | SL hits: {st['sl_hits']}")
    print(f"   Costs paid: {st['costs_paid']:.4f} | Swap paid: {st['swap_paid']:.4f}")
    print(f"   % Time Long: {np.mean(positions == 1) * 100:.1f}% | Short: {np.mean(positions == -1) * 100:.1f}%")

    print("\n🎉 Training Complete! The World Model has learned the Physics of the Market.")
    print("   Next: Implement MCTS to achieve true 'Stockfish' lookahead capability.")


if __name__ == "__main__":
    main()
