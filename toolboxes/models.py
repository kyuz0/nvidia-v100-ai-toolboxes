"""
Centralized model execution profiles for V100 benchmark and TUI runner.
V100 (sm_70, Volta) — 16 GB or 32 GB HBM2 per GPU.
  - FP16 only (no BF16)
  - bitsandbytes INT4 quantization support
  - CUDA Graphs supported
  - No FlashAttention v2 (requires sm_80+)
  - Triton attention + xFormers + PyTorch SDPA available
"""

GPU_UTIL = "0.90"

MODEL_TABLE = {
    # 1. Llama 3.1 8B Instruct (Native FP16)
    # V100-32GB: fits easily on 1 GPU. V100-16GB: tight fit on 1 GPU.
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "trust_remote": False,
        "valid_tp": [1, 2, 4],
        "max_num_seqs": "64",
        "max_tokens": "32768",
        "ctx": "65536"
    },

    # 2. Qwen 3.5 9B (Native FP16, GDN/DeltaNet hybrid architecture)
    # skip_profile_run=True: GDN Triton kernel hangs during profile_run on sm_70.
    # enforce_eager=True: skips torch.compile startup overhead.
    # gpu_util=0.65: hidden overhead at TP=2 is ~4.40 GiB per GPU:
    #   - SSM/DeltaNet state buffers (~0.97 GiB PyTorch)
    #   - NCCL + CUDA context (3.43 GiB non-PyTorch)
    #   free = 15.77*(1-gpu_util) - 4.40; at 0.72 → 8 MiB free (OOM), at 0.65 → 1.12 GiB
    # max_tokens=16384: causal_conv1d allocs (hidden/TP)×max_tokens×2B output buffer.
    #   At 32768 tokens: (4096/2)×32768×2 = 130 MiB → OOM. At 16384: ~64 MiB → fits.
    "Qwen/Qwen3.5-9B": {
        "trust_remote": True,
        "valid_tp": [1, 2, 4],
        "max_num_seqs": "64",
        "max_tokens": "16384",
        "ctx": "65536",
        "language_model_only": True,
        "enforce_eager": True,
        "gpu_util": "0.65",
        "skip_profile_run": True,
        "pytorch_expandable_segments": True  # fixes fragmentation OOM mid-run
    },

    # 3. Qwen 3.6 27B GPTQ 4bit — Volta-compatible (exllama kernel, not marlin)
    # skip_profile_run=True: gptq_gemm deadlocks during _dummy_run on sm_70 with TP>1.
    # gpu_util=0.65: skip_profile_run only tracks model weights (4.66 GiB) but misses
    #   hidden overhead (MoE routing, NCCL EP buffers, CUDA ctx ~3-4 GiB per GPU).
    #   Formula: free = 15.77*(1-gpu_util) - overhead
    #   At 0.75: ~52 MiB free (Triton 256 MiB cache flush fails).
    #   At 0.65: ~1.6-2 GiB free (enough for Triton JIT + kernel buffers).
    "btbtyler09/Qwen3.6-27B-GPTQ-4bit": {
        "trust_remote": True,
        "valid_tp": [1, 2, 4],
        "max_num_seqs": "32",
        "max_tokens": "16384",
        "ctx": "20480",
        "language_model_only": True,
        "enforce_eager": True,
        "gpu_util": "0.65",
        "quantization": "gptq",
        "disable_custom_allreduce": True,
        "nccl_p2p_disable": True,
        "skip_profile_run": True
    },

    # 4. Qwen 3.6 35B-A3B MoE GPTQ 4bit — Volta-compatible (exllama kernel, not marlin)
    # skip_profile_run=True: gptq_gemm deadlocks during _dummy_run on sm_70 with TP>1.
    # gpu_util=0.65: MoE model has ~3.89 GiB hidden overhead per GPU (expert routing,
    #   NCCL EP buffers, CUDA context) invisible to skip_profile_run.
    #   Observed at 0.75: 15.71 GiB in use, 52 MiB free → Triton 256 MiB flush OOM.
    #   At 0.65: ~1.63 GiB free → Triton JIT + kernel buffers fit.
    "palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4": {
        "trust_remote": True,
        "valid_tp": [1, 2, 4],
        "max_num_seqs": "32",
        "max_tokens": "16384",
        "ctx": "20480",
        "language_model_only": True,
        "enforce_eager": True,
        "gpu_util": "0.65",
        "quantization": "gptq",
        "disable_custom_allreduce": True,
        "nccl_p2p_disable": True,
        "skip_profile_run": True
    }
}

MODELS_TO_RUN = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen/Qwen3.5-9B",
    "btbtyler09/Qwen3.6-27B-GPTQ-4bit",
    "palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4"
]
