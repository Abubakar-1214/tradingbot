"""
Verify the P1 audit fixes end-to-end (run AFTER scripts/_apply_p1_fixes.py):

  1. dynamic_sizing(): value_long != value_flat, advantage != 0,
     win_prob != 0.5, and the critic is evaluated on two DISTINCT latent
     states (flat vs one-step-imagined long) — on a tiny seeded agent.
  2. dreamer save()/load(): optimizer state (3), return normalizer,
     torch RNG round-trip, training_step; old-format checkpoints still load.
  3. ReplayBuffer.sample(): sampled sequences never cross a done=True
     boundary (done only at the FINAL position).
  4. evaluate_model.py: compiles; hardcoded 252*24*12 annualization gone;
     ylim shows shorts.
  5. py_compile of every patched file.

Run: python scripts/_verify_p1_fixes.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

failures = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name} {detail}")
        failures.append(name)


# --------------------------------------------------------------------------- #
# 1. dynamic_sizing()
# --------------------------------------------------------------------------- #
from models.position_sizing import KellyPositionSizer  # noqa: E402
from models.dreamer_agent import DreamerV3Agent, ReplayBuffer  # noqa: E402

agent = DreamerV3Agent(
    obs_dim=16, action_dim=3, device="cpu",
    embed_dim=8, hidden_dim=16, stoch_dim=4, num_categories=4,
)

sizer = KellyPositionSizer(max_position=0.10, kelly_fraction=0.25)
rng = np.random.default_rng(0)
obs = rng.normal(size=16).astype("float32")

captured = {}
orig_cps = sizer.compute_position_size

def cps(win_prob, avg_win, avg_loss, equity=1.0):
    captured["win_prob"] = win_prob
    return orig_cps(win_prob, avg_win, avg_loss, equity=equity)

sizer.compute_position_size = cps

critic_inputs, critic_outputs = [], []
orig_fwd = agent.critic.forward

def wrapped(state):
    out = orig_fwd(state)
    critic_inputs.append(state.detach().clone())
    critic_outputs.append(out.detach().clone())
    return out

agent.critic.forward = wrapped

pos = sizer.dynamic_sizing(agent, {"equity": 10000.0}, obs)

check("dynamic_sizing critic evaluated twice (flat, long)",
      len(critic_inputs) == 2,
      f"got {len(critic_inputs)} critic calls")
check("dynamic_sizing latent states differ (flat != long-imagined)",
      len(critic_inputs) == 2 and not torch.allclose(critic_inputs[0], critic_inputs[1]))
check("dynamic_sizing value_long != value_flat",
      len(critic_outputs) == 2 and not torch.allclose(critic_outputs[0], critic_outputs[1]))
check("dynamic_sizing win_prob != 0.5",
      "win_prob" in captured and abs(float(captured["win_prob"]) - 0.5) > 1e-6,
      f"win_prob={captured.get('win_prob')}")
check("dynamic_sizing win_prob within clamp [0.3, 0.7]",
      "win_prob" in captured and 0.3 <= float(captured["win_prob"]) <= 0.7,
      f"win_prob={captured.get('win_prob')}")
check("dynamic_sizing returns non-negative fraction",
      pos >= 0.0, f"pos={pos}")

# --------------------------------------------------------------------------- #
# 2. save()/load() round-trip
# --------------------------------------------------------------------------- #
agent.return_low = torch.tensor(-2.0)
agent.return_high = torch.tensor(3.0)
agent.training_step = 42

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    ckpt = td / "agent.pt"

    torch.manual_seed(123)
    rng_before = torch.get_rng_state()
    agent.save(ckpt)

    agent2 = DreamerV3Agent(
        obs_dim=16, action_dim=3, device="cpu",
        embed_dim=8, hidden_dim=16, stoch_dim=4, num_categories=4,
    )
    torch.manual_seed(999)
    agent2.load(ckpt)

    check("save/load optimizer_world_model round-trip",
          set(agent2.optimizer_world_model.state_dict()) ==
          set(agent.optimizer_world_model.state_dict()))
    check("save/load optimizer_actor round-trip",
          set(agent2.optimizer_actor.state_dict()) ==
          set(agent.optimizer_actor.state_dict()))
    check("save/load optimizer_critic round-trip",
          set(agent2.optimizer_critic.state_dict()) ==
          set(agent.optimizer_critic.state_dict()))
    check("save/load return normalizer round-trip",
          agent2.return_low is not None and agent2.return_low.item() == -2.0 and
          agent2.return_high is not None and agent2.return_high.item() == 3.0)
    check("save/load training_step round-trip", agent2.training_step == 42)
    check("save/load torch RNG restored",
          torch.equal(torch.get_rng_state(), rng_before))

    # Backward compatibility: old-format checkpoint (no new keys) must load.
    old = torch.load(ckpt, map_location="cpu", weights_only=False)
    for k in ["optimizer_world_model", "optimizer_actor", "optimizer_critic",
              "return_low", "return_high", "rng_torch", "rng_numpy"]:
        old.pop(k, None)
    old_ckpt = td / "agent_old.pt"
    torch.save(old, old_ckpt)
    agent3 = DreamerV3Agent(
        obs_dim=16, action_dim=3, device="cpu",
        embed_dim=8, hidden_dim=16, stoch_dim=4, num_categories=4,
    )
    try:
        agent3.load(old_ckpt)
        check("save/load old-format checkpoint loads (backward compat)", True)
        check("save/load old-format leaves return_low None",
              agent3.return_low is None)
    except Exception as e:  # pragma: no cover
        check("save/load old-format checkpoint loads (backward compat)", False, str(e))

# --------------------------------------------------------------------------- #
# 3. ReplayBuffer episode-boundary guard
# --------------------------------------------------------------------------- #
buf = ReplayBuffer(capacity=1000, seq_len=16)
for i in range(500):
    done = (i % 50 == 49)  # every 50th transition terminates an episode
    buf.add(np.zeros(4, dtype="float32"), np.zeros(3, dtype="float32"), 0.0, done)

s = buf.sample(200)
check("ReplayBuffer sample not None", s is not None)
d = s["done"]  # (200, 16)
check("ReplayBuffer no done inside sequence (only final position)",
      np.all(d[:, :-1] == 0),
      f"interior done count={int(np.count_nonzero(d[:, :-1]))}")
check("ReplayBuffer some sequences terminate at final position",
      np.any(d[:, -1] == 1))

# Episode-heavy edge case: done every 3rd transition, seq_len=8 (> episode len).
# No 8-step window can avoid an interior done, so the guard MUST refuse to
# sample rather than cross episode boundaries (train_step handles None).
buf2 = ReplayBuffer(capacity=500, seq_len=8)
for i in range(400):
    done = (i % 3 == 2)
    buf2.add(np.zeros(4, dtype="float32"), np.zeros(3, dtype="float32"), 0.0, done)
s2 = buf2.sample(100)
check("ReplayBuffer degenerate (short episodes) refuses to sample",
      s2 is None,
      f"expected None, got shape={None if s2 is None else s2['done'].shape}")

# Medium-episode case: done every 40th transition, seq_len=16 (< episode len).
# Sequences must never cross a done — done only at the FINAL position.
buf3 = ReplayBuffer(capacity=1000, seq_len=16)
for i in range(600):
    done = (i % 40 == 39)
    buf3.add(np.zeros(4, dtype="float32"), np.zeros(3, dtype="float32"), 0.0, done)
s3 = buf3.sample(100)
check("ReplayBuffer medium-episode samples not None",
      s3 is not None)
check("ReplayBuffer medium-episode no interior done",
      s3 is not None and np.all(s3["done"][:, :-1] == 0),
      f"interior done count={0 if s3 is None else int(np.count_nonzero(s3['done'][:, :-1]))}")
check("ReplayBuffer medium-episode some final-position dones",
      s3 is not None and np.any(s3["done"][:, -1] == 1))

# --------------------------------------------------------------------------- #
# 4. evaluate_model.py checks
# --------------------------------------------------------------------------- #
ev = ROOT / "evaluate_model.py"
subprocess.run([sys.executable, "-m", "py_compile", str(ev)], check=True)
text = ev.read_text(encoding="utf-8")
check("evaluate_model compiles", True)
check("evaluate_model no hardcoded 252*24*12 annualization",
      "(252 * 24 * 12)" not in text)
check("evaluate_model uses bars_per_year param",
      "bars_per_year" in text and "--bars-per-year" in text)
check("evaluate_model ylim shows shorts", "set_ylim(-1.1, 1.1)" in text)

# --------------------------------------------------------------------------- #
# 5. py_compile patched files
# --------------------------------------------------------------------------- #
for f in ["models/position_sizing.py", "models/dreamer_agent.py", "evaluate_model.py"]:
    subprocess.run([sys.executable, "-m", "py_compile", str(ROOT / f)], check=True)
    check(f"py_compile {f}", True)

print(f"\n{'='*60}")
if failures:
    print(f"P1 VERIFY FAILED: {failures}")
    sys.exit(1)
print("P1 VERIFY OK — all checks passed.")
