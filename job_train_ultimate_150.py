#!/usr/bin/env python3
"""
================================================================================
⚡ LIGHTNING AI / GPU JOB RUNNER: DREAMERV3 ULTIMATE 150+
================================================================================
Interactive training script designed specifically for Lightning AI Studios & Jobs.
Eliminates Google Colab-specific code (Drive mount, temporary paths, files.download).
Retains:
  - Section 4 & 9: Real-time GPU detection & nvidia-smi monitoring (H100/H200/A100/T4)
  - Section 6: Main Training Execution with live interactive progress
  - Section 7: Smart Resume from checkpoint (auto-detects latest saved model)
  - Interactive prompts with RECOMMENDED defaults (press Enter to accept)
  - Non-interactive auto-fallback for Lightning Background Jobs
================================================================================
"""

import os
import sys
import glob
import subprocess
import shutil
from datetime import datetime

# Set up project path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "train", "dreamer_ultimate")


def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║     ⚡ LIGHTNING AI — DREAMERV3 ULTIMATE 150+ INTERACTIVE JOB RUNNER        ║
║         (Optimized for NVIDIA H100 / H200 / A100 / T4 Supercomputers)        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def check_gpu():
    """Detect GPU hardware and calculate hardware-specific recommendations."""
    gpu_info = {
        "available": False,
        "name": "CPU",
        "vram_gb": 0.0,
        "recommended_batch": 16,
        "time_per_1m_hrs": 120.0,
        "device": "cpu"
    }

    try:
        import torch
        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["device"] = "cuda"
            gpu_info["name"] = torch.cuda.get_device_name(0)
            gpu_info["vram_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

            name_upper = gpu_info["name"].upper()
            if "H200" in name_upper:
                gpu_info["recommended_batch"] = 1024
                gpu_info["time_per_1m_hrs"] = 0.5
            elif "H100" in name_upper:
                gpu_info["recommended_batch"] = 512
                gpu_info["time_per_1m_hrs"] = 0.7
            elif "A100" in name_upper:
                gpu_info["recommended_batch"] = 512 if gpu_info["vram_gb"] > 50 else 256
                gpu_info["time_per_1m_hrs"] = 3.0
            elif "V100" in name_upper:
                gpu_info["recommended_batch"] = 128
                gpu_info["time_per_1m_hrs"] = 8.0
            elif "T4" in name_upper:
                gpu_info["recommended_batch"] = 128
                gpu_info["time_per_1m_hrs"] = 14.0
            else:
                gpu_info["recommended_batch"] = 128
                gpu_info["time_per_1m_hrs"] = 10.0
    except Exception as e:
        print(f"[Warning] PyTorch CUDA check failed: {e}")

    return gpu_info


def run_nvidia_smi():
    """Section 9: Monitor GPU status using nvidia-smi."""
    if shutil.which("nvidia-smi"):
        print("\n" + "─"*78)
        print("📊 [Section 9] Current GPU Status (nvidia-smi):")
        print("─"*78)
        try:
            subprocess.run(["nvidia-smi"], check=False)
        except Exception:
            pass
        print("─"*78 + "\n")


from typing import Any, Callable


def prompt_user(prompt_text: str, default_value: Any, val_type: Callable[[str], Any] = str) -> Any:
    """
    Prompt user interactively. If user presses Enter without typing,
    default_value is automatically selected.
    If running in non-interactive environment (Background Job), defaults are used.
    """
    # Check if stdin is an interactive terminal
    if not sys.stdin.isatty():
        print(f"{prompt_text} -> [Non-interactive Job: Auto-selected: {default_value}]")
        return default_value

    try:
        user_input = input(prompt_text).strip()
        if not user_input:
            return default_value
        return val_type(user_input)
    except (EOFError, KeyboardInterrupt):
        print(f"\n[Auto-selected default]: {default_value}")
        return default_value


def scan_checkpoints():
    """Section 7: Search for existing checkpoints to allow resuming."""
    if not os.path.exists(CHECKPOINT_DIR):
        return []
    
    ckpts = glob.glob(os.path.join(CHECKPOINT_DIR, "*.pt"))
    # Sort by modification time
    ckpts.sort(key=os.path.getmtime)
    return ckpts


def main():
    print_banner()

    # =========================================================================
    # Step 1: Detect Hardware & Display GPU Status
    # =========================================================================
    print("🔍 Checking Hardware & Environment...")
    gpu = check_gpu()

    if gpu["available"]:
        print(f"  ✅ GPU Detected: {gpu['name']}")
        print(f"  💾 Total VRAM:   {gpu['vram_gb']:.2f} GB")
        print(f"  ⚡ Accelerator:  CUDA Enabled")
    else:
        print("  ⚠️  No GPU Detected. Running on CPU (Training will be slow).")

    # Run quick nvidia-smi monitor
    run_nvidia_smi()

    # =========================================================================
    # Step 2: Interactive Configuration (with Recommended Defaults)
    # =========================================================================
    print("⚙️  Configure Training Parameters (Press [ENTER] to accept Recommended):")
    print("─"*78)

    # 1. Training Steps
    print("Choose Training Scale:")
    print("  [1] Quick Pipeline Test   :     10,000 steps (~2-4 mins on H100)")
    print("  [2] Short Initial Learning:    100,000 steps (~10-15 mins on H100)")
    print("  [3] Full God Mode Model   :  1,000,000 steps (~1-1.5 hrs on H100) 🔥 [RECOMMENDED]")
    print("  (or type custom number of steps)")

    step_choice = prompt_user(
        "👉 Select Steps [1/2/3 or number] [Default: 3 (1,000,000 steps)]: ",
        default_value="3",
        val_type=str
    )

    if step_choice == "1":
        training_steps = 10_000
    elif step_choice == "2":
        training_steps = 100_000
    elif step_choice == "3":
        training_steps = 1_000_000
    else:
        try:
            training_steps = int(step_choice.replace(",", "").replace("_", ""))
        except ValueError:
            print("  ⚠️  Invalid input, falling back to 1,000,000 steps.")
            training_steps = 1_000_000

    # 2. Batch Size
    rec_batch = gpu["recommended_batch"]
    batch_prompt = f"👉 Batch Size [Default: {rec_batch} (Recommended for {gpu['name']})]: "
    batch_input = prompt_user(batch_prompt, default_value=rec_batch, val_type=int)
    batch_size = batch_input

    # 3. Base Timeframe
    tf_prompt = "👉 Base Timeframe (M5/M15/H1) [Default: M5 (Recommended)]: "
    base_tf = prompt_user(tf_prompt, default_value="M5", val_type=str).upper()
    if base_tf not in ["M5", "M15", "H1"]:
        base_tf = "M5"

    # 4. Checkpoint Frequency
    ckpt_freq_prompt = "👉 Checkpoint Frequency (Save every N steps) [Default: 10,000 (Recommended)]: "
    ckpt_freq_input = prompt_user(ckpt_freq_prompt, default_value="10,000", val_type=str)
    try:
        save_every = int(str(ckpt_freq_input).replace(",", "").replace("_", ""))
    except ValueError:
        save_every = 10_000

    # 5. Checkpoints & Resume (Section 7)
    checkpoints = scan_checkpoints()
    resume_checkpoint = None

    if checkpoints:
        print("\n" + "─"*78)
        print(f"📂 [Section 7] Found {len(checkpoints)} Existing Checkpoints in {CHECKPOINT_DIR}:")
        for cp in checkpoints[-3:]:
            size_mb = os.path.getsize(cp) / (1024 * 1024)
            mtime = datetime.fromtimestamp(os.path.getmtime(cp)).strftime('%Y-%m-%d %H:%M')
            print(f"   • {os.path.basename(cp)} ({size_mb:.1f} MB) - {mtime}")

        latest_ckpt = checkpoints[-1]
        resume_prompt = f"👉 Resume from latest checkpoint ({os.path.basename(latest_ckpt)})? [Y/n, Default: Y (Recommended)]: "
        resume_ans = prompt_user(resume_prompt, default_value="Y", val_type=str).strip().lower()

        if resume_ans in ["y", "yes", ""]:
            resume_checkpoint = latest_ckpt
            print(f"  ✅ Will resume from: {resume_checkpoint}")
        else:
            print("  🆕 Starting fresh training from step 0.")
    else:
        print("\n📁 No previous checkpoints found. Starting fresh training.")

    # =========================================================================
    # Step 3: Configuration Summary
    # =========================================================================
    time_rate = gpu.get("time_per_1m_hrs", 10.0)
    est_hours = (training_steps / 1_000_000) * time_rate

    print("\n" + "═"*78)
    print("📋 CONFIRMED TRAINING CONFIGURATION:")
    print("═"*78)
    print(f"  • Steps          : {training_steps:,}")
    print(f"  • Batch Size     : {batch_size}")
    print(f"  • Base Timeframe : {base_tf}")
    print(f"  • Checkpoint Save: Every {save_every:,} steps")
    print(f"  • Device         : {gpu['device']} ({gpu['name']})")
    print(f"  • Resume From    : {resume_checkpoint if resume_checkpoint else 'Fresh Start (0)'}")
    print(f"  • Est. Runtime   : ~{est_hours:.2f} hours on this machine")
    print(f"  • Output Folder  : {CHECKPOINT_DIR}")
    print("═"*78)

    confirm = prompt_user("\n🚀 Launch Training now? [Y/n, Default: Y]: ", default_value="Y", val_type=str)
    if confirm.strip().lower() not in ["y", "yes", ""]:
        print("❌ Training cancelled by user.")
        sys.exit(0)

    # =========================================================================
    # Step 4: Execute Training with Live Interactive Progress (Section 6)
    # =========================================================================
    print("\n" + "═"*78)
    print("🚀 [Section 6] STARTING TRAINING WITH INTERACTIVE PROGRESS")
    print("═"*78 + "\n")

    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "train", "train_ultimate_150.py"),
        "--steps", str(training_steps),
        "--batch-size", str(batch_size),
        "--device", gpu["device"],
        "--base-tf", base_tf,
        "--save-every", str(save_every)
    ]

    if resume_checkpoint:
        cmd.extend(["--resume", resume_checkpoint])

    start_time = datetime.now()
    try:
        # Run process and stream stdout directly to the terminal
        process = subprocess.run(cmd, check=True)
        elapsed = datetime.now() - start_time
        print("\n" + "═"*78)
        print(f"🎉 TRAINING COMPLETED SUCCESSFULLY! Total Time: {elapsed}")
        print("═"*78)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training process exited with error code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user. Any saved checkpoints in train/dreamer_ultimate/ remain safe.")

    # =========================================================================
    # Step 5: Post-Training Status & Checkpoints Summary (Section 9)
    # =========================================================================
    print("\n📊 Checking Final Saved Models...")
    final_ckpts = scan_checkpoints()
    if final_ckpts:
        print(f"✅ Total Checkpoints Saved: {len(final_ckpts)}")
        print(f"🏆 Latest Model Checkpoint: {final_ckpts[-1]}")
        print(f"💾 Size: {os.path.getsize(final_ckpts[-1]) / (1024*1024):.2f} MB")
        print("\n💡 Lightning AI Persistence Note:")
        print("   Aapki models Lightning Studio ki persistent storage mein mehfooz hain.")
        print("   Aap left sidebar se right-click karke direct download ya deploy kar sakte hain.")
    else:
        print("⚠️  No checkpoint files found in directory.")

    run_nvidia_smi()


if __name__ == "__main__":
    main()
