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

For the vLLM backend (powered by the jajmangold/vllm-sm70 community fork):
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
*   **No Flash Attention:** The V100 (sm_70) does not support FlashAttention v2 (requires Ampere sm_80+). The `-fa` flag in llama.cpp will not provide benefits on this hardware.
*   **No BF16:** The V100 does not support BF16. Always use FP16 (`--dtype half` in vLLM, default in llama.cpp).
*   **Multi-GPU:** For PCIe V100s, use `--split-mode layer` (default). For SXM2/NVLink V100s, `--split-mode tensor` can offer better performance due to the 300 GB/s NVLink 2.0 bandwidth.
*   **CUDA Version:** These toolboxes use CUDA 12.6.3 — the last CUDA release with full, non-deprecated sm_70 support. CUDA 13.0+ has removed V100 architecture support entirely.

## Build Automation
These toolboxes are automatically rebuilt every 4 hours via GitHub Actions if upstream `llama.cpp` has a new commit on its master branch.

## Benchmarks

*Benchmarks will be populated after running `benchmark/run_benchmarks.sh` on V100 hardware.*
