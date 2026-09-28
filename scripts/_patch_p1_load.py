"""
Fix torch.load weights_only for full training checkpoints (P1 verify failure).

PyTorch 2.6 changed torch.load() default to weights_only=True, which rejects
numpy RNG state stored in full training checkpoints (optimizer state + RNG).
Training checkpoints are trusted local artifacts, so load() must use
weights_only=False (the standard practice for RL/training checkpoints).

Run: python scripts/_patch_p1_load.py
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


# 1. dreamer_agent.py load(): allow trusted full training checkpoints
patch_file(
    ROOT / "models" / "dreamer_agent.py",
    "        checkpoint = torch.load(path, map_location=self.device)",
    "        # Full training checkpoints contain optimizer + numpy RNG state,\n"
    "        # which torch.load()'s default weights_only=True rejects.\n"
    "        # These are trusted local artifacts, so weights_only=False is\n"
    "        # the correct (standard) setting for training checkpoints.\n"
    "        checkpoint = torch.load(\n"
    "            path, map_location=self.device, weights_only=False\n"
    "        )",
)

# 2. verify script backward-compat probe
patch_file(
    ROOT / "scripts" / "_verify_p1_fixes.py",
    '    old = torch.load(ckpt, map_location="cpu")',
    '    old = torch.load(ckpt, map_location="cpu", weights_only=False)',
)

print("\nweights_only patches applied successfully.")
