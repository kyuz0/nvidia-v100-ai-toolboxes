#!/usr/bin/env bash
set -uo pipefail

MODEL_DIR="$(realpath ~/models)"
RESULTDIR="results"
mkdir -p "$RESULTDIR"

# Pick exactly one .gguf per model: either
#  - any .gguf without "-000*-of-" (single-file models)
#  - or the first shard "*-00001-of-*.gguf"
mapfile -t MODEL_PATHS < <(
  find "$MODEL_DIR" -type f -name '*.gguf' \
    \( -name '*-00001-of-*.gguf' -o -not -name '*-000*-of-*.gguf' \) \
    | sort
)

if (( ${#MODEL_PATHS[@]} == 0 )); then
  echo "❌ No models found under $MODEL_DIR – check your paths/patterns!"
  exit 1
fi

echo "Found ${#MODEL_PATHS[@]} model(s) to bench:"
for p in "${MODEL_PATHS[@]}"; do
  echo "  • $p"
done
echo

declare -A CMDS=(
  [v100]="toolbox run -c llama-v100-cuda -- /usr/local/bin/llama-bench"
  [vulkan]="toolbox run -c llama-v100-vulkan -- /usr/local/bin/llama-bench"
)

# Auto-detect VRAM per GPU to set thresholds
VRAM_PER_GPU_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
if [[ -z "$VRAM_PER_GPU_MB" ]]; then
  VRAM_PER_GPU_MB=16384  # Default to 16GB if detection fails
fi

# Calculate usable VRAM per GPU in bytes (~90% utilization)
USABLE_PER_GPU=$(( VRAM_PER_GPU_MB * 1024 * 1024 * 90 / 100 ))
echo "Detected ${VRAM_PER_GPU_MB} MiB VRAM per GPU (usable: ~$(( USABLE_PER_GPU / 1024 / 1024 / 1024 )) GiB)"
echo

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
  MODEL_NAME="$(basename "$MODEL_PATH" .gguf)"

  if [[ "$MODEL_PATH" == *"-00001-of-"* ]]; then
    # Multi-shard model: sum all shards
    DIR="$(dirname "$MODEL_PATH")"
    BASE="$(basename "$MODEL_PATH")"
    PATTERN="${BASE/-00001-of-/-*-of-}"
    # Pure bash addition to avoid `awk` scientific notation formatting bugs
    MODEL_SIZE=0
    for file in "$DIR"/$PATTERN; do
      if [[ -f "$file" ]]; then
        size=$(stat -c%s "$file")
        ((MODEL_SIZE+=size))
      fi
    done
  else
    # Single-file model
    MODEL_SIZE=$(stat -c%s "$MODEL_PATH")
  fi

  # Dynamically calculate GPU count needed based on detected VRAM
  if (( MODEL_SIZE > USABLE_PER_GPU * 3 )); then
    GPU_DEVICES="0,1,2,3"
    GPU_SUFFIX="__quad"
  elif (( MODEL_SIZE > USABLE_PER_GPU * 2 )); then
    GPU_DEVICES="0,1,2"
    GPU_SUFFIX="__triple"
  elif (( MODEL_SIZE > USABLE_PER_GPU )); then
    GPU_DEVICES="0,1"
    GPU_SUFFIX="__dual"
  else
    GPU_DEVICES="0"
    GPU_SUFFIX="__single"
  fi

  for ENV in "${!CMDS[@]}"; do
    CMD="${CMDS[$ENV]}"
    # Inject CUDA_VISIBLE_DEVICES before the executable
    CMD_EFFECTIVE="${CMD/-- /-- env CUDA_VISIBLE_DEVICES=$GPU_DEVICES }"

    # V100 does NOT support Flash Attention (requires sm_80+)
    # Run without -fa flag, unlike P100 which used -fa 1 (falling back anyway)
    EXTRA_ARGS=()

    for CTX in default longctx32768; do
      CTX_SUFFIX=""
      CTX_ARGS=()
      if [[ "$CTX" == longctx32768 ]]; then
        CTX_SUFFIX="__longctx32768"
        CTX_ARGS=( -p 2048 -n 32 -d 32768 )
        if [[ "$ENV" == *vulkan* ]]; then
          CTX_ARGS+=( -ub 512 )
        else
          CTX_ARGS+=( -ub 2048 )
        fi
      fi

      OUT="$RESULTDIR/${MODEL_NAME}__${ENV}${CTX_SUFFIX}${GPU_SUFFIX}.log"
      CTX_REPS=3
      if [[ "$CTX" == longctx* ]]; then
        CTX_REPS=1
      fi

      if [[ -s "$OUT" ]]; then
        echo "⏩ Skipping [${ENV}] ${MODEL_NAME}${CTX_SUFFIX:+ ($CTX_SUFFIX)}, log already exists at $OUT"
        continue
      fi

      FULL_CMD=( $CMD_EFFECTIVE -ngl 99 -m "$MODEL_PATH" "${EXTRA_ARGS[@]}" "${CTX_ARGS[@]}" -r "$CTX_REPS" )

      printf "\n▶ [%s] %s%s\n" "$ENV" "$MODEL_NAME" "${CTX_SUFFIX:+ $CTX_SUFFIX}"
      printf "  → log: %s\n" "$OUT"
      printf "  → cmd: %s\n\n" "${FULL_CMD[*]}"

      if ! "${FULL_CMD[@]}" >"$OUT" 2>&1; then
        status=$?
        echo "✖ ! [${ENV}] ${MODEL_NAME}${CTX_SUFFIX:+ $CTX_SUFFIX} failed (exit ${status})" >>"$OUT"
        echo "  * [${ENV}] ${MODEL_NAME}${CTX_SUFFIX:+ $CTX_SUFFIX} : FAILED"
      fi
    done
  done
done
