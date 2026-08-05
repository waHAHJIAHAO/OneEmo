swift export \
    --model_type qwen3_5 \
    --model /path/to/your/OneEmo/ckpts/inference_ckpts/sft/oneemo_p1_e5_20260525 \
    --adapters /path/to/your/resource/checkpoint-11000 \
    --output_dir /path/to/your/OneEmo/ckpts/inference_ckpts/grpo/oneemo \
    --merge_lora true