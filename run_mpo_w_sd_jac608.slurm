#!/usr/bin/env bash
#SBATCH --job-name=mpo_w_sd_run
#SBATCH --cluster=gpu
#SBATCH --partition=rtx6k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

set -euo pipefail

source /ix/cs2770_2026s/jac608/project/vllm_env/bin/activate

# Remove CUDA_VISIBLE_DEVICES — let SLURM handle it
# Also override GPU_NUMS in start_server.sh
export GPU_NUMS_OVERRIDE=0

cd /ix/cs2770_2026s/jac608/project
echo "Starting vLLM server..."
bash start_server.sh &
VLLM_PID=$!

echo "Waiting for vLLM to be ready..."
MAX_WAIT=300
ELAPSED=0
until curl -s http://localhost:13141/health > /dev/null 2>&1; do
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "vLLM failed to start within ${MAX_WAIT}s, exiting"
        kill $VLLM_PID
        exit 1
    fi
    echo "Still waiting... (${ELAPSED}s)"
done
echo "vLLM ready after ${ELAPSED}s"

echo "Starting MPO run..."
cd /ix/cs2770_2026s/jac608/project/MPO
bash main_mpo_w_sd_jac608.sh

echo "Done, shutting down vLLM"
kill $VLLM_PID
wait $VLLM_PID 2>/dev/null || true