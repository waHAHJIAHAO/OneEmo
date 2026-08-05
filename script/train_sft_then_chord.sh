#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ROLLOUT_HOST="${ROLLOUT_HOST:-127.0.0.1}"
ROLLOUT_PORT="${ROLLOUT_PORT:-8000}"

MODEL_DIR="${MODEL_DIR:-/path/to/your/OneEmo/ckpts/inference_ckpts/sft/oneemo_p1_e5_20260525}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/ckpts/training_ckpts/grpo_oneemo_chord}"
export VIDEMO_SENTENCE_BERT_MODEL="$REPO_ROOT/ckpts/all-mpnet-base-v2"

# CHORD is used here as an SFT anchor on the same task family as RL.
# The sample counts below are chosen so that each SFT dataset uses roughly the
# same sampling rate as its RL counterpart on the full dataset.
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
FORCE_QWENVL_VIDEO_READER=decord \
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
swift rlhf \
    --rlhf_type grpo \
    --model "$MODEL_DIR" \
    --dataset "$REPO_ROOT/datas/rl/dfew_msa.json#1000" \
              "$REPO_ROOT/datas/rl/dfew_mer.json#1200" \
              "$REPO_ROOT/datas/rl/mer2025ov.json#1000" \
              "$REPO_ROOT/datas/rl/mintrec.json#500" \
              "$REPO_ROOT/datas/rl/mintrec2.json#1200" \
              "$REPO_ROOT/datas/rl/mustard.json#160" \
              "$REPO_ROOT/datas/rl/urfunny.json#800" \
              "$REPO_ROOT/datas/rl/avamerg.json#2000" \
              "$REPO_ROOT/datas/rl/openr1psy.json#800" \
    --chord_sft_dataset "$REPO_ROOT/datas/sftnew/dfew_msa.json#3000" \
                        "$REPO_ROOT/datas/sftnew/dfew_mer.json#3600" \
                        "$REPO_ROOT/datas/sftnew/mercaptionplus.json#5000" \
                        "$REPO_ROOT/datas/sftnew/mintrec.json#1200" \
                        "$REPO_ROOT/datas/sftnew/mintrec2.json#3600" \
                        "$REPO_ROOT/datas/sftnew/mustard.json#387" \
                        "$REPO_ROOT/datas/sftnew/urfunny.json#1992" \
                        "$REPO_ROOT/datas/sftnew/avamerg.json#8000" \
                        "$REPO_ROOT/datas/sftnew/openr1psy.json#3000" \
    --chord_sft_per_device_train_batch_size 1 \
    --chord_mu_warmup_steps 500 \
    --chord_mu_decay_steps 6000 \
    --chord_mu_peak 0.5 \
    --chord_mu_valley 0.02 \
    --chord_enable_phi_function true \
    --load_from_cache_file true \
    --external_plugins "$REPO_ROOT/reward/reward_plugin.py" \
    --reward_funcs format process answer \
    --reward_weights 0.5 0.5 1.0 \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host "$ROLLOUT_HOST" \
    --vllm_server_port "$ROLLOUT_PORT" \
    --vllm_tensor_parallel_size 1 \
    --gradient_accumulation_steps 8 \
    --max_length 1536 \
    --max_completion_length 1024 \
    --overlong_filter true \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --dataset_num_proc 2 \
    --dataloader_num_workers 2 \
    --num_generations 8 \
    --temperature 0.7 \
    --top_p 0.9 \
    --epsilon 0.2 \
    --beta 0.06 \
    --tuner_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --torch_dtype bfloat16 \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --learning_rate 2e-6 \
    --lr_scheduler_type cosine \
    --new_special_tokens "$SCRIPT_DIR/task_tokens.txt" \
    --enable_thinking true \
    --output_dir "$OUTPUT_DIR" \
    --warmup_ratio 0.05 \
    --deepspeed zero2 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 20 \
    --log_completions true \
    --dynamic_sample true \
    --truncation_strategy delete \
    --overlong_filter true \
    --max_resample_times 5
