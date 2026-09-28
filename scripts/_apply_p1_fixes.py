"""
Apply the remaining P1 audit fixes with CRLF-safe byte-level patching
(edit_file cannot match Windows-written files in this repo).

Fixes applied:
  1. models/position_sizing.py KellyPositionSizer.dynamic_sizing():
     - RSSM.observe() returns FOUR values (h, z, prior_logits, posterior_logits);
       z is ALREADY the posterior sample.  The old code did a 2-tuple unpack
       and passed action=None (GRU concatenates z_prev+action).
     - value_long is now the critic value AFTER imagining the LONG action one
       step forward in the world model (rssm.imagine), so the advantage is a
       REAL quantity and win_prob != 0.5 (previously value_long = value_flat
       forced advantage == 0 and win_prob == sigmoid(0) == 0.5).
  2. models/dreamer_agent.py save()/load():
     - persist optimizer state (world_model/actor/critic), return normalizer
       (return_low/return_high), torch + numpy RNG states, training_step.
     - load() restores them (backward compatible via .get / membership checks).
  3. models/dreamer_agent.py ReplayBuffer.sample():
     - episode-boundary guard: sampled seq_len sequences never CROSS a done=True
       boundary (done may only appear at the FINAL position).  Vectorized via
       prefix-sum window counts (O(capacity), independent of batch_size).
  4. evaluate_model.py (repo root):
     - annualization parameterized via bars_per_year (default 72576 = M5),
       CLI --bars-per-year; no more hardcoded 252*24*12.
     - ylim(-1.1, 1.1) so short positions are visible in the position panel.

Run: python scripts/_apply_p1_fixes.py
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def patch_file(path: Path, anchor: str, replacement: str) -> None:
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    lf = text.replace("\r\n", "\n")
    n = lf.count(anchor)
    if n != 1:
        raise SystemExit(f"FAIL {path}: anchor found {n} times (expected 1)")
    out = lf.replace(anchor, replacement, 1)
    path.write_bytes(out.replace("\n", "\r\n").encode("utf-8"))
    print(f"[OK] patched {path.name} ({n} anchor occurrence)")


# --------------------------------------------------------------------------- #
# 1. position_sizing.py — dynamic_sizing()
# --------------------------------------------------------------------------- #
ps = ROOT / "models" / "position_sizing.py"

ps_anchor = """            # Get posterior state
            h, z_dist = agent.rssm.observe(embed, None, h, z)
            z = z_dist.sample()

            # Flatten z for critic
            z_flat = z.reshape(z.shape[0], -1)

            # Concatenate h and z for critic input
            state = torch.cat([h, z_flat], dim=-1)

            # Get action logits from actor
            action_logits = agent.actor(state)
            action_probs = torch.softmax(action_logits, dim=-1)

            # Get values for each action
            # For binary action (flat=0, long=1):
            value_flat = agent.critic(state)  # Value of current state
            # Estimate value if we go long (approximate)
            value_long = value_flat  # Simplified - in practice, simulate forward

            # Expected advantage
            advantage = value_long - value_flat

            # Convert to probability using sigmoid
            # Higher advantage = higher confidence = higher win probability
            win_prob = torch.sigmoid(advantage * 5).item()

            # Clamp to reasonable range
            win_prob = max(0.3, min(0.7, win_prob))"""

ps_replacement = """            # Posterior inference: RSSM.observe returns FOUR values
            # (h, z, prior_logits, posterior_logits); z is ALREADY the
            # posterior sample.  Passing action=None and re-sampling from a
            # distribution object was the P1 bug (wrong args + 2-tuple unpack).
            prev_action = getattr(agent, 'prev_action', None)
            if prev_action is None:
                prev_action = torch.zeros(1, agent.action_dim, device=agent.device)
                prev_action[0, 0] = 1.0  # flat
            h, z, _prior, _posterior = agent.rssm.observe(
                embed, prev_action, h, z
            )

            # Value of the CURRENT state (staying flat)
            state = agent.rssm.get_state(h, z)
            value_flat = agent.critic(state)

            # Value AFTER taking the LONG action: imagine one step forward in
            # the world model (prior dynamics), then critic at the new state.
            # Previously value_long = value_flat made advantage always 0 and
            # win_prob always 0.5 (P1 audit fix).
            long_action = torch.zeros(1, agent.action_dim, device=agent.device)
            long_action[0, 1] = 1.0  # long
            h_next, z_next, _prior_next = agent.rssm.imagine(long_action, h, z)
            state_long = agent.rssm.get_state(h_next, z_next)
            value_long = agent.critic(state_long)

            # Expected advantage of going long vs staying flat (REAL quantity)
            advantage = value_long - value_flat

            # Convert to probability using sigmoid
            # Higher advantage = higher confidence = higher win probability
            win_prob = torch.sigmoid(advantage * 5).item()

            # Clamp to reasonable range
            win_prob = max(0.3, min(0.7, win_prob))"""

patch_file(ps, ps_anchor, ps_replacement)

# --------------------------------------------------------------------------- #
# 2. dreamer_agent.py — save()/load() full training state
# --------------------------------------------------------------------------- #
da = ROOT / "models" / "dreamer_agent.py"

da_save_anchor = """    def save(self, path):
        \"\"\"Save agent\"\"\"
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
        \"\"\"Load agent\"\"\"
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.rssm.load_state_dict(checkpoint['rssm'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.reward_predictor.load_state_dict(checkpoint['reward_predictor'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.slow_critic.load_state_dict(checkpoint.get('slow_critic', checkpoint['critic']))
        self.training_step = checkpoint.get('training_step', 0)
        print(f\"Loaded checkpoint from step {self.training_step}\")"""

da_save_replacement = """    def save(self, path):
        \"\"\"Save agent — FULL training state (P1 audit fix).

        Persists network weights, optimizer state (world_model/actor/critic),
        the return normalizer (5th/95th percentile EMA used by
        _normalize_returns), torch + numpy RNG states and the training step so
        a resumed run continues deterministically from the checkpoint.
        \"\"\"
        torch.save({
            'encoder': self.encoder.state_dict(),
            'rssm': self.rssm.state_dict(),
            'decoder': self.decoder.state_dict(),
            'reward_predictor': self.reward_predictor.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'slow_critic': self.slow_critic.state_dict(),
            'optimizer_world_model': self.optimizer_world_model.state_dict(),
            'optimizer_actor': self.optimizer_actor.state_dict(),
            'optimizer_critic': self.optimizer_critic.state_dict(),
            'return_low': self.return_low,
            'return_high': self.return_high,
            'training_step': self.training_step,
            'rng_torch': torch.get_rng_state(),
            'rng_numpy': np.random.get_state(),
        }, path)

    def load(self, path):
        \"\"\"Load agent — restores the FULL training state (P1 audit fix).

        Backward compatible: checkpoints written by the old implementation
        (networks + training_step only) load with defaults for the new keys.
        \"\"\"
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.rssm.load_state_dict(checkpoint['rssm'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.reward_predictor.load_state_dict(checkpoint['reward_predictor'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.slow_critic.load_state_dict(checkpoint.get('slow_critic', checkpoint['critic']))
        if 'optimizer_world_model' in checkpoint:
            self.optimizer_world_model.load_state_dict(checkpoint['optimizer_world_model'])
        if 'optimizer_actor' in checkpoint:
            self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor'])
        if 'optimizer_critic' in checkpoint:
            self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic'])
        self.return_low = checkpoint.get('return_low')
        self.return_high = checkpoint.get('return_high')
        self.training_step = checkpoint.get('training_step', 0)
        if 'rng_torch' in checkpoint:
            torch.set_rng_state(checkpoint['rng_torch'])
        if 'rng_numpy' in checkpoint:
            np.random.set_state(checkpoint['rng_numpy'])
        print(f\"Loaded checkpoint from step {self.training_step}\")"""

patch_file(da, da_save_anchor, da_save_replacement)

# --------------------------------------------------------------------------- #
# 3. dreamer_agent.py — ReplayBuffer.sample() episode-boundary guard
# --------------------------------------------------------------------------- #
rb_anchor = """        # Logical index 0 is the oldest transition
        oldest = (self.ptr - self.size) % self.capacity
        starts = np.random.randint(0, self.size - self.seq_len, size=batch_size)
        idx = (oldest + starts[:, None] + np.arange(self.seq_len)[None, :]) % self.capacity

        return {
            'obs': self.obs[idx],
            'action': self.action[idx],
            'reward': self.reward[idx],
            'done': self.done[idx],
        }"""

rb_replacement = """        # Logical index 0 is the oldest transition.  Build logical-order done
        # flags for the valid region (ring order -> chronological order).
        oldest = (self.ptr - self.size) % self.capacity
        if oldest <= self.ptr - 1:  # no ring wrap in the valid region
            done_logical = self.done[oldest:self.ptr]
        else:
            done_logical = np.concatenate(
                [self.done[oldest:], self.done[:self.ptr]]
            )

        # Episode-boundary guard (P1 audit fix): a sampled sequence of length
        # seq_len never CROSSES a done=True boundary — a done flag may only
        # appear at the FINAL position of the sequence (the transition that
        # terminated the episode).  Valid starts are found vectorized with a
        # prefix-sum window count: start s is valid iff no done appears in
        # [s, s+seq_len-2].  O(capacity) regardless of batch_size.
        window = self.seq_len - 1
        prefix = np.zeros(self.size + 1, dtype=np.int64)
        np.cumsum(done_logical, out=prefix[1:])
        if window > 0:
            counts = prefix[window:] - prefix[:-window]
        else:
            counts = np.zeros(self.size, dtype=np.int64)
        max_start = self.size - self.seq_len
        valid = np.flatnonzero(counts[:max_start + 1] == 0)
        if len(valid) == 0:
            # Degenerate fallback (episodes shorter than the sequence): sample
            # starts whose preceding transition did not terminate an episode.
            preceding_done = np.concatenate([[0.0], done_logical[:-1]])
            valid = np.flatnonzero(preceding_done[:max_start + 1] == 0.0)
        if len(valid) == 0:
            valid = np.arange(max_start + 1)

        starts = valid[np.random.randint(0, len(valid), size=batch_size)]
        idx = (oldest + starts[:, None] + np.arange(self.seq_len)[None, :]) % self.capacity

        return {
            'obs': self.obs[idx],
            'action': self.action[idx],
            'reward': self.reward[idx],
            'done': self.done[idx],
        }"""

patch_file(da, rb_anchor, rb_replacement)

# --------------------------------------------------------------------------- #
# 4. evaluate_model.py — bars_per_year param + ylim showing shorts
# --------------------------------------------------------------------------- #
ev = ROOT / "evaluate_model.py"

patch_file(
    ev,
    "def evaluate_model(agent, env, timestamps):",
    "def evaluate_model(agent, env, timestamps, bars_per_year=72576):",
)

ev_ann_anchor = """    # Annualized metrics (assuming 252 trading days)
    days = len(equity_curve) / (252 * 24 * 12)  # Convert 5-min bars to years
    annual_return = ((equity_curve[-1] ** (1 / days)) - 1) * 100 if days > 0 else 0

    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24 * 12)"""

ev_ann_replacement = """    # Annualized metrics (timeframe-correct bars-per-year; P1 audit fix —
    # previously hardcoded 252*24*12 which silently assumed 5-minute bars)
    days = len(equity_curve) / bars_per_year
    annual_return = ((equity_curve[-1] ** (1 / days)) - 1) * 100 if days > 0 else 0

    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(bars_per_year)"""

patch_file(ev, ev_ann_anchor, ev_ann_replacement)

patch_file(
    ev,
    "    axes[2].set_ylim(-0.1, 1.1)",
    "    axes[2].set_ylim(-1.1, 1.1)  # long AND short visible (P1 audit fix)",
)

ev_arg_anchor = """    parser.add_argument('--save-plot', type=str, default='evaluation_results.png',
                       help='Path to save results plot')

    args = parser.parse_args()"""

ev_arg_replacement = """    parser.add_argument('--save-plot', type=str, default='evaluation_results.png',
                       help='Path to save results plot')
    parser.add_argument('--bars-per-year', type=float, default=None,
                       help='Bars per trading year for annualization '
                            '(default: 72576 = 252*24*12 for M5 bars; '
                            'use 8760 for H1, 365 for D1)')

    args = parser.parse_args()"""

patch_file(ev, ev_arg_anchor, ev_arg_replacement)

ev_call_anchor = """    # ========== EVALUATE ==========
    metrics, equity_curve, positions, dates = evaluate_model(agent, env, timestamps_eval)"""

ev_call_replacement = """    # ========== EVALUATE ==========
    bars_per_year = args.bars_per_year or 72576  # M5 default; override per timeframe
    metrics, equity_curve, positions, dates = evaluate_model(
        agent, env, timestamps_eval, bars_per_year=bars_per_year)"""

patch_file(ev, ev_call_anchor, ev_call_replacement)

print("\nAll P1 patches applied successfully.")
