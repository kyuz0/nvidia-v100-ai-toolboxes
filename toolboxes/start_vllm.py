#!/usr/bin/env python3
"""
vLLM Launcher for NVIDIA Tesla V100 (sm_70 / Volta)
====================================================
Interactive TUI to select and serve models via the official vllm/vllm-openai
v0.18.1 image. Supports both V100-16GB and V100-32GB.

V100 characteristics:
  - CUDA Graphs supported — enforce-eager is optional
  - No FlashAttention v2 (requires Ampere sm_80+)
  - No BF16 — always --dtype half (FP16)
  - bitsandbytes INT4 quantization supported
  - VLLM_USE_V1=0 for stability (V0 engine)
"""
import sys
import os
import shutil
import tempfile
import subprocess
import time
from pathlib import Path

# Add directory to path to import config
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, "/opt")

try:
    from models import MODEL_TABLE, MODELS_TO_RUN, GPU_UTIL
except ImportError:
    print("Error: Could not import models.py config. Ensure models.py is in the same directory or /opt.")
    sys.exit(1)

HOST = os.getenv("HOST", "0.0.0.0")
PORT = os.getenv("PORT", "8000")


def detect_vram_per_gpu():
    """Auto-detect VRAM per GPU in GB. Returns 32 or 16."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        if res.returncode == 0:
            vram_mb = int(res.stdout.strip().split("\n")[0].strip())
            return 32 if vram_mb > 20000 else 16
    except (FileNotFoundError, ValueError):
        pass
    return 16


VRAM_PER_GPU_GB = detect_vram_per_gpu()


def check_dependencies():
    if not shutil.which("dialog"):
        print("Error: 'dialog' is required. Please install it (apt-get install dialog).")
        sys.exit(1)


def detect_gpus():
    """Detect NVIDIA GPUs via nvidia-smi."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        if res.returncode == 0:
            lines = [l.strip() for l in res.stdout.strip().split("\n") if l.strip()]
            return len(lines)
    except FileNotFoundError:
        pass
    return 1


def run_dialog(args):
    """Runs dialog and returns stderr (selection)."""
    with tempfile.NamedTemporaryFile(mode="w+") as tf:
        cmd = ["dialog"] + args
        try:
            subprocess.run(cmd, stderr=tf, check=True)
            tf.seek(0)
            return tf.read().strip()
        except subprocess.CalledProcessError:
            return None


def kill_gpu_zombies():
    """Kill any lingering processes holding CUDA/GPU contexts (zombie vLLM runs etc.)."""
    print("  Killing GPU zombie processes...", end="", flush=True)
    try:
        # Get PIDs of processes using any /dev/nvidia device
        res = subprocess.run(
            ["fuser", "/dev/nvidia0", "/dev/nvidia1", "/dev/nvidia2", "/dev/nvidia3",
             "/dev/nvidiactl", "/dev/nvidia-uvm"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
        )
        pids = set(res.stdout.split())
        own_pid = str(os.getpid())
        pids.discard(own_pid)
        if pids:
            subprocess.run(["kill", "-9"] + list(pids),
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(3)  # allow CUDA contexts to fully release
            print(f" Killed {len(pids)} process(es).")
        else:
            print(" None found.")
    except FileNotFoundError:
        # fuser not available — fallback: kill vllm/python procs by name
        subprocess.run("pgrep -f 'vllm serve' | xargs -r kill -9",
                       shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)
        print(" Done (name-based kill).")
    except Exception as e:
        print(f" Warning: {e}")


def nuke_vllm_cache():
    """Removes vLLM cache directory."""
    for cache_dir, label in [
        (Path.home() / ".cache" / "vllm", "vLLM"),
        (Path.home() / ".triton" / "cache", "Triton"),
    ]:
        if cache_dir.exists():
            try:
                print(f"  Clearing {label} cache at {cache_dir}...", end="", flush=True)
                subprocess.run(["rm", "-rf", str(cache_dir)], check=True)
                print(" Done.")
            except Exception as e:
                print(f" Failed: {e}")


def configure_and_launch(model_idx, gpu_count):
    model_id = MODELS_TO_RUN[model_idx]
    config = MODEL_TABLE[model_id]

    valid_tps = config.get("valid_tp", [1])
    max_tp = max(valid_tps) if valid_tps else 1

    current_tp = min(gpu_count, max_tp)
    current_seqs = int(config.get("max_num_seqs", "32"))
    current_ctx = int(config.get("ctx", "8192"))
    current_util = float(config.get("gpu_util", GPU_UTIL))
    use_eager = config.get("enforce_eager", False)
    clear_cache = False

    name = model_id.split("/")[-1]

    while True:
        cache_status = "YES" if clear_cache else "NO"
        eager_status = "YES" if use_eager else "NO"

        menu_args = [
            "--clear", "--backtitle", f"V100-{VRAM_PER_GPU_GB}GB vLLM Launcher (GPUs: {gpu_count} detected)",
            "--title", f"Configuration: {name}",
            "--menu", "Customize Launch Parameters:", "22", "65", "9",
            "1", f"Tensor Parallelism:   {current_tp}",
            "2", f"Concurrent Requests:  {current_seqs}",
            "3", f"Context Length:       {current_ctx}",
            "4", f"GPU Utilization:      {current_util}",
            "5", f"Erase vLLM Cache:     {cache_status}",
            "6", f"Force Eager Mode:     {eager_status}",
            "7", "LAUNCH SERVER"
        ]

        choice = run_dialog(menu_args)
        if not choice:
            return False

        if choice == "1":
            tp_menu = []
            for tp_val in valid_tps:
                if tp_val <= gpu_count:
                    tp_menu.extend([str(tp_val), f"TP={tp_val} GPU(s)"])
            if tp_menu:
                new_tp = run_dialog([
                    "--title", "Tensor Parallelism",
                    "--menu", "Select Tensor Parallelism (TP) Size:", "12", "50", "4"
                ] + tp_menu)
                if new_tp:
                    current_tp = int(new_tp)
        elif choice == "2":
            new_seqs = run_dialog(["--title", "Concurrent Requests", "--inputbox", "Max Concurrent Requests:", "10", "40", str(current_seqs)])
            if new_seqs: current_seqs = int(new_seqs)
        elif choice == "3":
            new_ctx = run_dialog(["--title", "Context Length", "--inputbox", "Max Model Context Length:", "10", "40", str(current_ctx)])
            if new_ctx: current_ctx = int(new_ctx)
        elif choice == "4":
            new_util = run_dialog(["--title", "GPU Utilization", "--inputbox", "GPU Memory Utilization (0.1 - 1.0):", "10", "40", str(current_util)])
            if new_util: current_util = float(new_util)
        elif choice == "5":
            clear_cache = not clear_cache
        elif choice == "6":
            use_eager = not use_eager
        elif choice == "7":
            break

    # Build Command
    subprocess.run(["clear"])
    print("Pre-launch cleanup:")
    kill_gpu_zombies()  # always purge zombie CUDA contexts before launching
    if clear_cache:
        nuke_vllm_cache()

    cmd = [
        "vllm", "serve", model_id,
        "--host", HOST,
        "--port", PORT,
        "--tensor-parallel-size", str(current_tp),
        "--max-num-seqs", str(current_seqs),
        "--max-model-len", str(current_ctx),
        "--gpu-memory-utilization", str(current_util),
        "--dtype", "half"
    ]

    if config.get("trust_remote"): cmd.append("--trust-remote-code")
    if use_eager: cmd.append("--enforce-eager")
    if config.get("language_model_only"): cmd.append("--language-model-only")
    # GPTQ: explicitly force exllama kernel (avoids auto-selecting gptq_marlin which needs sm_75+)
    if config.get("quantization"): cmd.extend(["--quantization", config["quantization"]])
    # GPTQ on PCIe V100: disable vLLM's custom allreduce (shm_broadcast) — it deadlocks on Volta
    if config.get("disable_custom_allreduce"): cmd.append("--disable-custom-all-reduce")

    env = os.environ.copy()
    # NOTE: VLLM_USE_V1 is not recognized in v0.18.1 — V1 engine is mandatory
    env["PYTHONNOUSERSITE"] = "1"
    # GPTQ on PCIe V100: disable NCCL P2P — no NVLink means peer-access fails silently and stalls
    if config.get("nccl_p2p_disable"):
        env["NCCL_P2P_DISABLE"] = "1"
        env["NCCL_SHM_DISABLE"] = "1"
    # GPTQ on sm_70: gptq_gemm deadlocks during profiling forward pass — skip it
    if config.get("skip_profile_run"):
        env["VLLM_SKIP_PROFILE_RUN"] = "1"

    print("\n" + "=" * 60)
    print(f" Launching: {name}")
    print(f" Hardware:  V100-{VRAM_PER_GPU_GB}GB x {gpu_count}")
    print(f" Config:    TP={current_tp} | Seqs={current_seqs} | Ctx={current_ctx} | Util={current_util} | Eager={use_eager}")
    if config.get("quantization"):
        skip_str = " | SkipProfile=YES" if config.get("skip_profile_run") else ""
        print(f" Quant:     {config['quantization']} | CustomAllReduce=OFF | NCCL_P2P=OFF | NCCL_SHM=OFF{skip_str}")
    if clear_cache:
        print(f" Action:    Clearing vLLM Cache (~/.cache/vllm)")
    print(f" Command:   {' '.join(cmd)}")
    print("=" * 60 + "\n")

    os.execvpe("vllm", cmd, env)


def main():
    check_dependencies()
    gpu_count = detect_gpus()

    while True:
        menu_items = []
        for i, m_id in enumerate(MODELS_TO_RUN):
            name = m_id.split("/")[-1]
            menu_items.extend([str(i), name])

        choice = run_dialog([
            "--clear", "--backtitle", f"V100-{VRAM_PER_GPU_GB}GB vLLM Launcher (GPUs: {gpu_count})",
            "--title", "Select Model",
            "--menu", "Choose a model to serve:", "18", "60", "10"
        ] + menu_items)

        if not choice:
            subprocess.run(["clear"])
            print("Selection cancelled.")
            sys.exit(0)

        configure_and_launch(int(choice), gpu_count)


if __name__ == "__main__":
    main()
