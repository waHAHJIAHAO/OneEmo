#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
swift sft \
    --model $REPO_ROOT/ckpts/inference_ckpts/sft/oneemo_p1_e5_20260618 \
    --tuner_type lora \
    --dataset "$REPO_ROOT/datas/sftnew/dfew_msa.json#2000" \
              "$REPO_ROOT/datas/sftnew/dfew_mer.json#3000" \
              "$REPO_ROOT/datas/sftnew/merr_fine.json#3000" \
              "$REPO_ROOT/datas/sftnew/mercaptionplus.json#10000" \
              "$REPO_ROOT/datas/sftnew/mintrec.json#1200" \
              "$REPO_ROOT/datas/sftnew/mintrec2.json#3500" \
              "$REPO_ROOT/datas/sftnew/mustard.json#380" \
              "$REPO_ROOT/datas/sftnew/urfunny.json#1900" \
              "$REPO_ROOT/datas/sftnew/avamerg.json#10000" \
              "$REPO_ROOT/datas/sftnew/openr1psy.json#6000" \
    --load_from_cache_file true \
    --dataset_shuffle true \
    --torch_dtype bfloat16 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 5e-6 \
    --lr_scheduler_type constant \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --gradient_accumulation_steps 4 \
    --new_special_tokens './task_tokens.txt' \
    --enable_thinking true \
    --output_dir "$REPO_ROOT/ckpts/training_ckpts/sft_oneemo_p2" \
    --warmup_ratio 0.05 \
    --dataset_num_proc 2 \
    --dataloader_num_workers 2 \
    --save_strategy epoch \
    --deepspeed zero2