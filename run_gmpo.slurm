#!/usr/bin/env bash
#SBATCH --account=cs2770_2026s
#SBATCH --job-name=gmpo
#SBATCH --cluster=gpu
#SBATCH --partition=rtx6k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bkb45@pitt.edu

set -euo pipefail

source /ix/cs2770_2026s/jac608/project/vllm_env/bin/activate

# Remove CUDA_VISIBLE_DEVICES — let SLURM handle it
# Also override GPU_NUMS in start_server.sh
export GPU_NUMS_OVERRIDE=0

# # Persist model caches across jobs to avoid re-downloading weights.
# export HF_HOME=/ix/cs2770_2026s/jac608/project/.hf_cache
# export TRANSFORMERS_CACHE=/ix/cs2770_2026s/jac608/project/.hf_cache/transformers
# export VLLM_DOWNLOAD_DIR=/ix/cs2770_2026s/jac608/project/.hf_cache/vllm
# mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$VLLM_DOWNLOAD_DIR"
# export DIFFUSERS_LOCK_PATH=/ix/cs2770_2026s/jac608/project/.hf_cache/diffusers_generator.lock
# : > "$DIFFUSERS_LOCK_PATH"

cd /ix/cs2770_2026s/jac608/project
echo "Allocated GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Starting vLLM server..."
bash start_server.sh &
VLLM_PID=$!

echo "Waiting for vLLM to be ready..."
MAX_WAIT=900
ELAPSED=0
HEALTH_URL="http://localhost:13141/health"
until curl -s "$HEALTH_URL" > /dev/null 2>&1; do
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "vLLM failed to start within ${MAX_WAIT}s (health check: ${HEALTH_URL}), exiting"
        kill $VLLM_PID
        exit 1
    fi
    echo "Still waiting... (${ELAPSED}s)"
done
echo "vLLM ready after ${ELAPSED}s"

echo "Starting MPO run..."
cd /ix/cs2770_2026s/jac608/project/MPO
bash main_gmpo.sh

echo "Done, shutting down vLLM"
kill $VLLM_PID
wait $VLLM_PID 2>/dev/null || true