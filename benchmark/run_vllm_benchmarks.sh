#!/usr/bin/env bash
set -euo pipefail

# Host-side wrapper to run vLLM throughput benchmarks inside the vllm-v100 toolbox.
# Usage: ./run_vllm_benchmarks.sh [--tp 1 2 4]

echo "🚀 Starting vLLM Benchmarks inside the vllm-v100 toolbox..."
toolbox run -c vllm-v100 -- run-vllm-bench "$@"
echo "✅ Benchmarks completed. Results are saved in 'benchmark_results/'"
