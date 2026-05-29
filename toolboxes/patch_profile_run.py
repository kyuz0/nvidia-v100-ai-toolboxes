#!/usr/bin/env python3
"""
Apply the VLLM_SKIP_PROFILE_RUN patch to a live vLLM v0.18.1 container.
Run this INSIDE the container (as root or with write access to site-packages).

Usage:
    python3 patch_profile_run.py          # apply patch
    python3 patch_profile_run.py --check  # verify patch is already applied
    python3 patch_profile_run.py --undo   # revert to original
"""
import sys
import pathlib
import argparse
import shutil
import datetime

TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py"
)

BACKUP = TARGET.with_suffix(".py.orig")

OLD = (
    "    def profile_run(self) -> None:\n"
    "        # Profile with multimodal encoder & encoder cache."
)

NEW = (
    "    def profile_run(self) -> None:\n"
    "        # sm_70 / Volta workaround: VLLM_SKIP_PROFILE_RUN=1 bypasses the\n"
    "        # profiling forward pass that deadlocks with the gptq_gemm kernel.\n"
    "        import os as _os\n"
    "        if _os.environ.get('VLLM_SKIP_PROFILE_RUN') == '1':\n"
    "            logger.warning(\n"
    "                'VLLM_SKIP_PROFILE_RUN=1: skipping profiling forward pass. '\n"
    "                'KV-cache size is estimated from gpu_memory_utilization only '\n"
    "                '(sm_70/Volta workaround for gptq_gemm deadlock).'\n"
    "            )\n"
    "            return\n"
    "        # Profile with multimodal encoder & encoder cache."
)


def check(txt: str) -> str:
    if NEW in txt:
        return "patched"
    if OLD in txt:
        return "original"
    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Check patch status and exit")
    parser.add_argument("--undo",  action="store_true", help="Revert to original")
    args = parser.parse_args()

    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found. Are you inside the vllm-v100 container?")
        sys.exit(1)

    txt = TARGET.read_text()
    status = check(txt)

    # ── CHECK ──────────────────────────────────────────────────────────────────
    if args.check:
        print(f"Status: {status}")
        if status == "patched":
            print("✓ Patch is applied. Set VLLM_SKIP_PROFILE_RUN=1 before launching.")
        elif status == "original":
            print("✗ Patch NOT applied. Run without --check to apply.")
        else:
            print("? Neither original anchor nor patched anchor found — inspect manually.")
        sys.exit(0 if status == "patched" else 1)

    # ── UNDO ───────────────────────────────────────────────────────────────────
    if args.undo:
        if BACKUP.exists():
            shutil.copy2(BACKUP, TARGET)
            print(f"Restored from {BACKUP}")
        elif status == "patched":
            TARGET.write_text(txt.replace(NEW, OLD, 1))
            print("Reverted patch in-place (no backup found, used string replacement).")
        else:
            print("Nothing to undo — file is already original or unknown state.")
        sys.exit(0)

    # ── APPLY ──────────────────────────────────────────────────────────────────
    if status == "patched":
        print("Patch already applied — nothing to do.")
        print("Set VLLM_SKIP_PROFILE_RUN=1 before launching vLLM.")
        sys.exit(0)

    if status != "original":
        print("ERROR: Patch anchor not found. The vLLM version may have changed.")
        print(f"Expected to find this string in {TARGET}:")
        print(repr(OLD))
        sys.exit(1)

    # Backup original
    if not BACKUP.exists():
        shutil.copy2(TARGET, BACKUP)
        print(f"Backed up original → {BACKUP}")

    TARGET.write_text(txt.replace(OLD, NEW, 1))
    print(f"✓ Patched {TARGET}")
    print()
    print("To use:")
    print("  export VLLM_SKIP_PROFILE_RUN=1")
    print("  vllm serve btbtyler09/Qwen3.6-27B-GPTQ-4bit \\")
    print("    --tensor-parallel-size 4 --max-model-len 20480 \\")
    print("    --gpu-memory-utilization 0.85 --dtype half \\")
    print("    --quantization gptq --disable-custom-all-reduce \\")
    print("    --enforce-eager --language-model-only --trust-remote-code")
    print()
    print("Watch for: 'VLLM_SKIP_PROFILE_RUN=1: skipping profiling forward pass'")
    print("in the worker logs — that confirms the patch is active.")


if __name__ == "__main__":
    main()
