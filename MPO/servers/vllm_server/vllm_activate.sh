#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

echo "VLLM_API_KEY: ${VLLM_API_KEY:-}"
ulimit -n 51200

GPU_NUMS=(${GPU_NUMS_OVERRIDE:-0})

PORT_SUFFIX=1

TENSOR_PARALLEL_SIZE="${#GPU_NUMS[@]}"

# CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPU_NUMS[*]}")"
# export CUDA_VISIBLE_DEVICES
# echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

export NCCL_SOCKET_FAMILY=AF_INET

VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.20}"
VLLM_MAX_LEN="${VLLM_MAX_LEN:-8192}"
echo VLLM_MAX_LEN: $VLLM_MAX_LEN

ENFORCE_EAGER=(--enforce-eager)
if [ "${VLLM_SKIP_ENFORCE_EAGER:-0}" = "1" ]; then
  ENFORCE_EAGER=()
fi

vllm serve \
$MODEL_NAME \
--dtype auto \
--port 1314$PORT_SUFFIX \
--tensor-parallel-size $TENSOR_PARALLEL_SIZE \
--gpu-memory-utilization "$VLLM_GPU_MEM_UTIL" \
--max_model_len "$VLLM_MAX_LEN" \
"${ENFORCE_EAGER[@]}" \
--limit-mm-per-prompt '{"image": 2, "video": 2}' \
--max-num-seqs 1 \
--trust-remote-code