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
    # 1. Llama 3.1 8B Instruct
    # V100-32GB: fits easily on 1 GPU. V100-16GB: tight fit on 1 GPU.
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "trust_remote": False,
        "valid_tp": [1],
        "max_num_seqs": "64",
        "max_tokens": "32768",
        "ctx": "65536"
    },

    # 2. Qwen 3.5 9B (Native FP16)
    "Qwen/Qwen3.5-9B": {
        "trust_remote": True,
        "valid_tp": [1],
        "max_num_seqs": "64",
        "max_tokens": "32768",
        "ctx": "65536",
        "language_model_only": True
    },

    # 3. Qwen 3.6 27B AWQ INT4 — tight fit on 1x V100-32GB
    "cyankiwi/Qwen3.6-27B-AWQ-INT4": {
        "trust_remote": True,
        "valid_tp": [1],
        "max_num_seqs": "32",
        "max_tokens": "16384",
        "ctx": "20480",
        "language_model_only": True,
        "enforce_eager": True,
        "gpu_util": "0.95"
    },

    # 4. Qwen 3.6 35B-A3B AWQ (MoE, only 3B active params per token)
    "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit": {
        "trust_remote": True,
        "valid_tp": [1],
        "max_num_seqs": "32",
        "max_tokens": "16384",
        "ctx": "20480",
        "language_model_only": True
    },

    # 5. Gemma 4 26B-A4B AWQ (MoE, 4B active params per token)
    "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit": {
        "trust_remote": True,
        "valid_tp": [1],
        "max_num_seqs": "32",
        "max_tokens": "16384",
        "ctx": "20480",
        "language_model_only": True,
        "enforce_eager": True
    },

    # 6. Gemma 4 31B AWQ (Dense)
    "cyankiwi/gemma-4-31B-it-AWQ-4bit": {
        "trust_remote": True,
        "valid_tp": [1],
        "max_num_seqs": "32",
        "max_tokens": "4096",
        "ctx": "8192",
        "language_model_only": True,
        "enforce_eager": True,
        "gpu_util": "0.98"
    }
}

MODELS_TO_RUN = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen/Qwen3.5-9B",
    "cyankiwi/Qwen3.6-27B-AWQ-INT4",
    "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit",
    "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit",
    "cyankiwi/gemma-4-31B-it-AWQ-4bit"
]
