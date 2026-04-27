#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../vllm_env/bin/activate"

set -a
[[ -f "${SCRIPT_DIR}/.env" ]] && source "${SCRIPT_DIR}/.env"
set +a

BASE_MODEL='Qwen2.5-VL-7B'
OPTIM_MODEL='Qwen2.5-VL-7B'
MM_GENERATOR_MODEL='diffusers-flux-schnell'

ulimit -n 65535

METHOD=mpo
EXP_NAME=mpo

TASK="cuckoo"
BUDGET_PER_PROMPT=100

TRAIN_SIZE=86
TEST_SIZE=86

DATA_DIR="${SCRIPT_DIR}/../datasets"

LOG_DIR="./logs/$BASE_MODEL/$OPTIM_MODEL/$MM_GENERATOR_MODEL/${EXP_NAME}/${TASK}"

cd "$SCRIPT_DIR" || exit 1
python main.py \
    --data_dir "$DATA_DIR" \
    --task_name $TASK \
    --train_size $TRAIN_SIZE \
    --test_size $TEST_SIZE \
    --log_dir $LOG_DIR \
    --base_model_name $BASE_MODEL \
    --base_model_port 13141 \
    --optim_model_name $OPTIM_MODEL \
    --optim_model_temperature 0.0 \
    --optim_model_port 13141 \
    --mm_generator_model_name $MM_GENERATOR_MODEL \
    --search_method $METHOD \
    --iteration 5 \
    --beam_width 3 \
    --model_responses_num 3 \
    --seed 42 \
    --budget_per_prompt $BUDGET_PER_PROMPT \
    --evaluation_method bayes-ucb \
    --bayes_prior_strength 10 \
    --diffusers_num_inference_steps 4 \
    --diffusers_model_id /ix/cs2770_2026s/jac608/project/models/flux-schnell \