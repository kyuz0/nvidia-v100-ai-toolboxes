#!/usr/bin/env python3
"""
vLLM Throughput Benchmark for NVIDIA Tesla V100 (sm_70 / Volta)
================================================================
Adapted from the MI50 benchmark (run_vllm_bench_mi50.py).

V100-specific:
  - CUDA Graphs supported — enforce-eager per-model config
  - --dtype half (no BF16 on Volta)
  - VLLM_USE_V1=0 (V0 engine for stability)
"""
import subprocess, time, json, sys, os, requests, argparse
from pathlib import Path

# Add directory to path to import config
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, "/opt")

try:
    from models import GPU_UTIL, MODEL_TABLE, MODELS_TO_RUN
except ImportError:
    print("Error: Could not import models.py config. Ensure models.py is in the same directory or /opt.")
    sys.exit(1)

PORT     = 8000
HOST     = "127.0.0.1"

# THROUGHPUT CONFIG
OFF_NUM_PROMPTS      = 500
OFF_FORCED_OUTPUT    = "512"
DEFAULT_BATCH_TOKENS = "8192"

RESULTS_DIR = Path("benchmark_results")
RESULTS_DIR.mkdir(exist_ok=True)


def log(msg): print(f"\n[BENCH] {msg}")

def get_gpu_count():
    """Detect NVIDIA GPUs via nvidia-smi."""
    env_var = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if env_var:
        return len(env_var.split(","))
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

def kill_vllm():
    subprocess.run("pgrep -f 'vllm serve' | xargs -r kill -9",
                   shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(5)


def get_dataset():
    data_path = Path("ShareGPT_V3_unfiltered_cleaned_split.json")
    if data_path.exists():
        if data_path.stat().st_size > 100_000_000:
            return str(data_path)
        else:
            log("Found corrupted/incomplete ShareGPT dataset. Re-downloading...")
            data_path.unlink()

    log("Downloading ShareGPT dataset...")
    url = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
    try:
        r = requests.get(url, stream=True, timeout=15)
        r.raise_for_status()
        tmp_path = data_path.with_suffix(".tmp")
        with open(tmp_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        tmp_path.rename(data_path)
        return str(data_path)
    except Exception as e:
        log(f"WARNING: ShareGPT download failed ({e}). using RANDOM.")
        return None

def wait_for_server(url, process, timeout=600):
    start = time.time()
    while time.time() - start < timeout:
        if process.poll() is not None:
            log(f"CRITICAL: Server died! Ret: {process.returncode}")
            return False
        try:
            if requests.get(f"{url}/v1/models", timeout=2).status_code == 200:
                log("Server ready. Stabilizing...")
                time.sleep(5)
                return True
        except: pass
        time.sleep(2)
    return False

def get_model_args(model, tp_size):
    config = MODEL_TABLE.get(model, {"max_num_seqs": "32"})

    util = config.get("gpu_util", GPU_UTIL)

    cmd = [
        "--model", model,
        "--gpu-memory-utilization", util,
        "--dtype", "half",
        "--tensor-parallel-size", str(tp_size),
        "--max-num-seqs", config["max_num_seqs"]
    ]

    if "ctx" in config: cmd.extend(["--max-model-len", config["ctx"]])
    if config.get("trust_remote"): cmd.append("--trust-remote-code")
    if config.get("enforce_eager"): cmd.append("--enforce-eager")
    if config.get("language_model_only"): cmd.append("--language-model-only")
    # GPTQ: force exllama kernel explicitly (not gptq_marlin which requires sm_75+)
    if config.get("quantization"): cmd.extend(["--quantization", config["quantization"]])
    # GPTQ on PCIe V100: vLLM's custom allreduce (shm_broadcast) deadlocks on Volta
    if config.get("disable_custom_allreduce"): cmd.append("--disable-custom-all-reduce")
    return cmd

def run_throughput(model, tp_size, output_dir=RESULTS_DIR):
    if tp_size not in MODEL_TABLE[model]["valid_tp"]: return

    model_safe = model.replace("/", "_")
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    output_file = output_dir_path / f"{model_safe}_tp{tp_size}_throughput.json"

    if output_file.exists():
        log(f"SKIP {model} (TP={tp_size}) — result exists")
        return

    dataset_path = get_dataset()
    dataset_args = ["--dataset-name", "sharegpt", "--dataset-path", dataset_path] if dataset_path else ["--input-len", "1024"]

    batch_tokens = MODEL_TABLE[model].get("max_tokens", DEFAULT_BATCH_TOKENS)

    log(f"START {model} (TP={tp_size}) [Batch: {batch_tokens}]...")
    kill_vllm()

    cmd = ["vllm", "bench", "throughput"] + get_model_args(model, tp_size)
    cmd.extend([
        "--num-prompts", str(OFF_NUM_PROMPTS),
        "--max-num-batched-tokens", str(batch_tokens),
        "--output-len", OFF_FORCED_OUTPUT,
        "--output-json", str(output_file),
        "--disable-log-stats"
    ])
    cmd.extend(dataset_args)

    env = os.environ.copy()
    # NOTE: VLLM_USE_V1 not recognized in v0.18.1 — V1 engine is mandatory
    env["PYTHONNOUSERSITE"] = "1"
    # GPTQ on PCIe V100: no NVLink → NCCL P2P stalls; force net transport
    config = MODEL_TABLE.get(model, {})
    if config.get("nccl_p2p_disable"):
        env["NCCL_P2P_DISABLE"] = "1"
        env["NCCL_SHM_DISABLE"] = "1"
    # GPTQ on sm_70: gptq_gemm deadlocks during profiling forward pass — skip it
    if config.get("skip_profile_run"):
        env["VLLM_SKIP_PROFILE_RUN"] = "1"
    # Hybrid models (Qwen3.5-9B, Qwen3.6): DeltaNet/GDN activations fragment the
    # CUDA allocator pool mid-run → expandable segments let PyTorch serve large
    # requests from non-contiguous reserved pages instead of hard-crashing.
    if config.get("pytorch_expandable_segments"):
        env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        log(f"ERROR: Failed {model} (exit code {e.returncode})")
    except Exception as e:
        log(f"ERROR: Failed {model}: {type(e).__name__}: {e}")

def print_summary(tps):
    print(f"\n{'MODEL':<40} | {'TP':<2} | {'tok/s':<14}")
    print("-" * 65)

    for m in MODELS_TO_RUN:
        msafe = m.replace("/", "_")
        name_cell = m.split('/')[-1]

        for tp in tps:
            if tp not in MODEL_TABLE[m]["valid_tp"]: continue

            output_file = RESULTS_DIR / f"{msafe}_tp{tp}_throughput.json"

            try:
                if output_file.exists():
                    d = json.loads(output_file.read_text())
                    val = f"{d.get('tokens_per_second', 0):.1f}"
                else:
                    val = "N/A"
            except: val = "N/A"

            print(f"{name_cell:<40} | {tp:<2} | {val:<14}")
    print("-" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="vLLM throughput benchmark for NVIDIA V100 (Volta)"
    )
    parser.add_argument("--tp", type=int, nargs="+", default=[1])
    args = parser.parse_args()

    gpu_count = get_gpu_count()
    log(f"Detected {gpu_count} NVIDIA GPU(s)")

    # Set default TP sizes dynamically if not explicitly specified by user
    if args.tp == [1] and "--tp" not in sys.argv:
        if gpu_count >= 4:
            args.tp = [2, 4]
        elif gpu_count >= 2:
            args.tp = [2]
        else:
            args.tp = [1]

    valid_tp_args = [t for t in args.tp if t <= gpu_count]
    if not valid_tp_args:
        log(f"Requested TP={args.tp} but only {gpu_count} GPU(s) detected. Nothing to run.")
        sys.exit(0)

    # Clear the vLLM torch_compile cache (compiled graphs go stale between runs).
    # Do NOT clear the Triton kernel cache (~/.triton/cache): it stores autotuned
    # configs for the GDN/DeltaNet Triton kernels; rebuilding it costs 4-6 minutes
    # per model on first inference. Only clear it manually after a Triton upgrade.
    vllm_cache = Path.home() / ".cache" / "vllm" / "torch_compile_cache"
    if vllm_cache.exists():
        subprocess.run(["rm", "-rf", str(vllm_cache)], check=False)
        log(f"Cleared vLLM compile cache: {vllm_cache}")

    kill_vllm()
    for tp in valid_tp_args:
        for m in MODELS_TO_RUN:
            run_throughput(m, tp)

    print_summary(valid_tp_args)
