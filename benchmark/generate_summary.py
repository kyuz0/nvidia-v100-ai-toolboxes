#!/usr/bin/env python3
import os
import json
import re
from pathlib import Path

def parse_llama_log(filepath):
    results = {}
    size = "-"
    gpu_count = "-"
    
    if not filepath.exists():
        return size, gpu_count, results
        
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        
    # Find GPU count from lines like:
    # "ggml_cuda_init: found 2 CUDA devices (Total VRAM: 65076 MiB):"
    gpu_match = re.search(r"found (\d+) CUDA devices", content)
    if gpu_match:
        gpu_count = gpu_match.group(1)
        
    lines = content.splitlines()
    header = []
    for line in lines:
        if "| model" in line:
            header = [p.strip() for p in line.split("|")][1:-1]
        elif line.startswith("|") and "---" not in line and header:
            parts = [p.strip() for p in line.split("|")][1:-1]
            if len(parts) == len(header):
                row = dict(zip(header, parts))
                test_name = row.get('test', '')
                tps = row.get('t/s', '')
                if test_name and tps:
                    results[test_name] = tps
                if 'size' in row and row['size']:
                    size = row['size']
                    
    return size, gpu_count, results

def get_gpu_count_from_suffix(suffix):
    mapping = {
        "single": "1",
        "dual": "2",
        "triple": "3",
        "quad": "4"
    }
    return mapping.get(suffix, "-")

def clean_model_name(name):
    # Remove shard suffix
    name = re.sub(r"-0000\d-of-0000\d", "", name)
    return name

def generate_llama_tables(results_dir):
    if not results_dir.exists():
        return f"Llama.cpp results directory not found: {results_dir}\n"
        
    # Structure to hold llama.cpp data
    # key: (model, backend) -> { 'size': ..., 'gpu_count': ..., 'pp512': ..., 'tg128': ..., 'pp32k': ..., 'tg32k': ... }
    data = {}
    
    # Read files
    for filepath in results_dir.glob("*.log"):
        fname = filepath.name
        # e.g., Qwen3.5-35B-A3B-UD-Q4_K_XL__v100__longctx32768__dual.log
        # or Qwen3.5-35B-A3B-UD-Q4_K_XL__v100__dual.log
        parts = fname.replace(".log", "").split("__")
        if len(parts) < 2:
            continue
            
        model = parts[0]
        backend = parts[1]
        
        is_long = "longctx" in fname
        gpu_suffix = parts[-1]
        
        size, parsed_gpu_count, results = parse_llama_log(filepath)
        gpu_count = parsed_gpu_count if parsed_gpu_count != "-" else get_gpu_count_from_suffix(gpu_suffix)
        
        key = (model, backend)
        if key not in data:
            data[key] = {
                "size": "-",
                "gpu_count": gpu_count,
                "pp512": "-",
                "tg128": "-",
                "pp32k": "-",
                "tg32k": "-"
            }
            
        if size != "-":
            data[key]["size"] = size
            
        if gpu_count != "-":
            data[key]["gpu_count"] = gpu_count
            
        for test, ts in results.items():
            if "pp512" in test:
                data[key]["pp512"] = ts
            elif "tg128" in test:
                data[key]["tg128"] = ts
            elif "pp2048" in test:
                data[key]["pp32k"] = ts
            elif "tg32" in test:
                data[key]["tg32k"] = ts

    # Sort keys
    sorted_keys = sorted(data.keys(), key=lambda k: (k[0], k[1]))
    
    lines = []
    lines.append("### Llama.cpp Benchmarks (NGL=99)")
    lines.append("All tests run with `NGL=99` and Flash Attention enabled (`-fa 1`). (32k = PP2048 @ d32768, TG32 @ d32768)\n")
    
    lines.append("#### Prompt Processing (PP) Throughput")
    lines.append("| Model | Size | GPUs | Backend | PP512 | PP(32k) |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for key in sorted_keys:
        row = data[key]
        # Skip if all measurements are empty
        if row["pp512"] == "-" and row["pp32k"] == "-":
            continue
        cleaned_model = clean_model_name(key[0])
        lines.append(f"| {cleaned_model} | {row['size']} | {row['gpu_count']} | {key[1]} | {row['pp512']} | {row['pp32k']} |")
        
    lines.append("\n#### Text Generation (TG) Throughput")
    lines.append("| Model | Size | GPUs | Backend | TG128 | TG(32k) |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for key in sorted_keys:
        row = data[key]
        # Skip if all measurements are empty
        if row["tg128"] == "-" and row["tg32k"] == "-":
            continue
        cleaned_model = clean_model_name(key[0])
        lines.append(f"| {cleaned_model} | {row['size']} | {row['gpu_count']} | {key[1]} | {row['tg128']} | {row['tg32k']} |")
    lines.append("")
    
    return "\n".join(lines)

def generate_vllm_table(results_dir):
    if not results_dir.exists():
        return f"vLLM results directory not found: {results_dir}\n"
        
    lines = []
    lines.append("### vLLM Throughput")
    lines.append("| Model | TP | Requests | Total Tokens | Tokens/sec | Requests/sec | Elapsed (sec) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    
    files = list(results_dir.glob("*.json"))
    files.sort()
    
    for file in files:
        # e.g. Qwen_Qwen2.5-14B-Instruct_tp4_throughput.json
        name_parts = file.stem.split("_tp")
        model = name_parts[0]
        tp = name_parts[1].split("_")[0] if len(name_parts) > 1 else "1"
        
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        requests = data.get("num_requests", "-")
        tokens = data.get("total_num_tokens", "-")
        tps = data.get("tokens_per_second", 0)
        rps = data.get("requests_per_second", 0)
        elapsed = data.get("elapsed_time", 0)
        
        lines.append(f"| {model} | {tp} | {requests} | {tokens} | {tps:.2f} | {rps:.4f} | {elapsed:.2f} |")
    
    lines.append("")
    return "\n".join(lines)

def update_main_readme(main_readme_path, benchmark_markdown):
    if not main_readme_path.exists():
        return
        
    with open(main_readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    marker = "## Benchmarks"
    
    # Let's check if the marker exists.
    # If it does, we replace the section. If not, we append it.
    if marker in content:
        # Keep everything before the marker, and replace the rest
        parts = content.split(marker)
        new_content = parts[0] + marker + "\n\n" + benchmark_markdown
    else:
        new_content = content.rstrip() + "\n\n" + marker + "\n\n" + benchmark_markdown
        
    with open(main_readme_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f"Updated main README at {main_readme_path}")

def main():
    base_dir = Path(__file__).parent
    
    results_llamacpp = base_dir / "results_llamacpp"
    results_vllm = base_dir / "results_vllm"
    
    llama_output = generate_llama_tables(results_llamacpp)
    vllm_output = generate_vllm_table(results_vllm)
    
    benchmark_content = llama_output + "\n" + vllm_output
    
    markdown_output = "# Benchmarks (NVIDIA Tesla V100)\n\n"
    markdown_output += benchmark_content
    
    # Print to stdout
    print(markdown_output)
    
    # Save to benchmark/README.md
    readme_path = base_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(markdown_output)
    print(f"Saved summary to {readme_path}")
    
    # Update main project README.md
    main_readme_path = base_dir.parent / "README.md"
    update_main_readme(main_readme_path, benchmark_content)

if __name__ == "__main__":
    main()
