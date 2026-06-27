# NVIDIA V100 (sm_70) AI Toolboxes

This repository provides automatically built Toolbox (`containertoolbx.org`) images with `llama.cpp` natively compiled and optimized for NVIDIA Tesla V100 GPUs (Volta architecture, Compute Capability 7.0).

> [!NOTE]
> The V100 is the last NVIDIA architecture fully supported by CUDA 12.x. These builds use **CUDA 12.6.3** (the last release with first-class sm_70 support) and compile `llama.cpp` with the `-DCMAKE_CUDA_ARCHITECTURES=70` flag. CUDA 13.0+ has removed sm_70 offline compilation support.

## Getting Started

### 1. Prerequisites
You need a system with `nvidia-container-toolkit` installed and configured so that your container runtime (Docker or Podman) can access the GPUs. You also need `toolbox` installed.

If you don't have `nvidia-container-toolkit` installed (e.g., you get a `command not found` error), please follow the [official installation instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for your distribution.

For Podman, once installed, make sure the Container Device Interface (CDI) is generated:
```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```

### 2. Create the Toolbox
Create a new toolbox utilizing the published image. Since Toolbox automatically handles GPU passthrough when `nvidia-container-toolkit` is present, no extra device flags are needed:

For the standard CUDA backend:
```bash
toolbox create -c llama-v100-cuda \
  --image docker.io/kyuz0/nvidia-v100-ai-toolboxes:latest
```

Alternatively, if you want to use the Vulkan backend (built on Fedora 43):
```bash
toolbox create -c llama-v100-vulkan \
  --image docker.io/kyuz0/nvidia-v100-ai-toolboxes:vulkan
```

For the vLLM backend (powered by the official vllm/vllm-openai:v0.18.1 base):
```bash
toolbox create -c vllm-v100 \
  --image docker.io/kyuz0/nvidia-v100-ai-toolboxes:vllm
```

### 3. Enter the Toolbox
```bash
toolbox enter llama-v100-cuda
```
*Note: The toolboxes resolve common UID 1000 conflicts, meaning your host user ID and home directories will map seamlessly into the container.*

### 4. Run Inference

**With Llama.cpp:**
You can run `llama-server` or `llama-cli` directly since the binaries are located in `/usr/local/bin/`.

Example command using the first GPU:
```bash
llama-server -m ~/models/Llama-3-8B-Instruct.Q4_K_M.gguf -ngl 99 -c 4096 --port 8080
```

**With vLLM:**
If you entered the `vllm-v100` toolbox, you can start the OpenAI-compatible vLLM server:

```bash
vllm serve ~/models/Qwen2.5-7B-Instruct --dtype half --max-model-len 4096
```

## Performance & Optimization Tips

*   **VRAM Limits:** V100 GPUs come in 16GB and 32GB variants. Ensure your model and KV cache fit entirely within VRAM. Q4_K_M and Q5_K_M quantizations work well for most models.
*   **CUDA Graphs:** Unlike the P100, the V100 fully supports CUDA graphs. `llama.cpp` utilizes these automatically for batch size 1, significantly improving token generation speed.
*   **Flash Attention:** While standard FlashAttention-2 (e.g. standard PyTorch packages) requires Ampere sm_80+, `llama.cpp`'s custom Flash Attention kernels (`-fa 1`) DO support Volta sm_70 and are fully utilized for optimal performance.
*   **No BF16:** The V100 does not support BF16. Always use FP16 (`--dtype half` in vLLM, default in llama.cpp).
*   **Multi-GPU:** For PCIe V100s, use `--split-mode layer` (default). For SXM2/NVLink V100s, `--split-mode tensor` can offer better performance due to the 300 GB/s NVLink 2.0 bandwidth.
*   **CUDA Version:** These toolboxes use CUDA 12.6.3 — the last CUDA release with full, non-deprecated sm_70 support. CUDA 13.0+ has removed V100 architecture support entirely.

## Build Automation
These toolboxes are automatically rebuilt every 4 hours via GitHub Actions if upstream `llama.cpp` has a new commit on its master branch.

## Benchmarks

### Llama.cpp Benchmarks (NGL=99, FA=1)
All tests run with `NGL=99` and `FA=1`. (32k = PP2048 @ d32768, TG32 @ d32768)

#### Prompt Processing (PP) Throughput
| Model | Size | GPUs | Backend | PP512 | PP(32k) |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-122B-A10B-Q3_K_M | 52.54 GiB | 4 | v100 | 487.02 ± 2.02 | 464.36 ± 0.00 |
| Qwen3.5-35B-A3B-UD-Q4_K_XL | 20.70 GiB | 2 | v100 | 811.16 ± 3.03 | 824.86 ± 0.00 |
| Qwen3.5-35B-A3B-UD-Q8_K_XL | 45.33 GiB | 4 | v100 | 941.60 ± 9.86 | 866.05 ± 0.00 |
| Qwen3.6-27B-UD-Q4_K_XL | 16.39 GiB | 2 | v100 | 852.18 ± 2.52 | 622.78 ± 0.00 |
| Qwen3.6-27B-UD-Q8_K_XL | 32.89 GiB | 3 | v100 | 1012.61 ± 3.09 | 629.22 ± 0.00 |
| Qwen3.6-35B-A3B-UD-Q4_K_XL | 20.81 GiB | 2 | v100 | 804.11 ± 2.48 | 837.46 ± 0.00 |
| Qwen3.6-35B-A3B-UD-Q8_K_XL | 35.80 GiB | 3 | v100 | 670.61 ± 4.03 | 797.49 ± 0.00 |
| gemma-4-26B-A4B-it-UD-Q4_K_XL | 15.90 GiB | 2 | v100 | 1615.92 ± 8.70 | 1318.48 ± 0.00 |
| gemma-4-26B-A4B-it-UD-Q8_K_XL | 25.94 GiB | 2 | v100 | 1430.12 ± 11.53 | 1258.70 ± 0.00 |
| gemma-4-E4B-it-UD-Q8_K_XL | 8.05 GiB | 1 | v100 | 4659.53 ± 54.68 | 2177.93 ± 0.00 |
| gpt-oss-20b-mxfp4 | 11.27 GiB | 1 | v100 | 2521.60 ± 5.84 | 2043.74 ± 0.00 |

#### Text Generation (TG) Throughput
| Model | Size | GPUs | Backend | TG128 | TG(32k) |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-122B-A10B-Q3_K_M | 52.54 GiB | 4 | v100 | 48.67 ± 1.74 | 46.41 ± 0.00 |
| Qwen3.5-35B-A3B-UD-Q4_K_XL | 20.70 GiB | 2 | v100 | 102.09 ± 0.08 | 91.47 ± 0.00 |
| Qwen3.5-35B-A3B-UD-Q8_K_XL | 45.33 GiB | 4 | v100 | 84.62 ± 1.28 | 76.42 ± 0.00 |
| Qwen3.6-27B-UD-Q4_K_XL | 16.39 GiB | 2 | v100 | 34.13 ± 0.01 | 28.14 ± 0.00 |
| Qwen3.6-27B-UD-Q8_K_XL | 32.89 GiB | 3 | v100 | 21.62 ± 0.01 | 18.77 ± 0.00 |
| Qwen3.6-35B-A3B-UD-Q4_K_XL | 20.81 GiB | 2 | v100 | 103.76 ± 0.07 | 92.97 ± 0.00 |
| Qwen3.6-35B-A3B-UD-Q8_K_XL | 35.80 GiB | 3 | v100 | 96.82 ± 2.25 | 87.23 ± 0.00 |
| gemma-4-26B-A4B-it-UD-Q4_K_XL | 15.90 GiB | 2 | v100 | 99.62 ± 0.04 | 87.07 ± 0.00 |
| gemma-4-26B-A4B-it-UD-Q8_K_XL | 25.94 GiB | 2 | v100 | 90.42 ± 0.06 | 79.44 ± 0.00 |
| gemma-4-E4B-it-UD-Q8_K_XL | 8.05 GiB | 1 | v100 | 87.01 ± 0.04 | 77.12 ± 0.00 |
| gpt-oss-20b-mxfp4 | 11.27 GiB | 1 | v100 | 158.57 ± 0.11 | 134.14 ± 0.00 |

### vLLM Throughput
| Model | TP | Requests | Total Tokens | Tokens/sec | Requests/sec | Elapsed (sec) |
| --- | --- | --- | --- | --- | --- | --- |
| meta-llama_Meta-Llama-3.1-8B-Instruct | 2 | 500 | 361361 | 1625.33 | 2.2489 | 222.33 |
| meta-llama_Meta-Llama-3.1-8B-Instruct | 4 | 500 | 361361 | 2483.35 | 3.4361 | 145.51 |
| btbtyler09_Qwen3.6-27B-GPTQ-4bit | 2 | 500 | 367735 | 124.77 | 0.1696 | 2947.36 |
| btbtyler09_Qwen3.6-27B-GPTQ-4bit | 4 | 500 | 367735 | 194.65 | 0.2647 | 1889.22 |
| palmfuture_Qwen3.6-35B-A3B-GPTQ-Int4 | 2 | 500 | 367735 | 17.49 | 0.0238 | 21021.41 |
| palmfuture_Qwen3.6-35B-A3B-GPTQ-Int4 | 4 | 500 | 367735 | 304.50 | 0.4140 | 1207.68 |
| Qwen_Qwen3.5-9B | 4 | 500 | 367735 | 1438.30 | 1.9556 | 255.67 |
