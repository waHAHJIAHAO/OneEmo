<p align="center">
  <img src="assets/logo.png" width="180" alt="OneEmo logo">
</p>

<h1 align="center">OneEmo: A Unified Multimodal Reasoning Model for Emotion Perception, Understanding, and Interaction</h1>

<p align="center">
    <a href="https://arxiv.org/pdf/2603.02123">
    <img src='https://img.shields.io/badge/Paper-Arxiv-orange' alt='Paper PDF'></a>
    <a href="https://huggingface.co/Jiaha0Hu4ng/OneEmo">
    <img src='https://img.shields.io/badge/Model-HuggingFace-yellow' alt='Model'></a>
    <a href='https://huggingface.co/datasets/Jiaha0Hu4ng/EmoWorld-130K'">
    <img src='https://img.shields.io/badge/Dataset-HuggingFace-yellow' alt='Dataset'></a>
</p>


## Overview

OneEmo targets video emotion intelligence, unifying emotion perception, emotion understanding, and emotion interaction within a single multimodal reasoning model. It covers sentiment analysis, basic emotion recognition, open-vocabulary emotion recognition, intention recognition, and humor understanding. Compared with models of similar scale, OneEmo achieves SOTA results on eight emotion tasks. The training pipeline consists of two core components:

1. **EmoWorld-130K**: Distills domain knowledge from expert models across multiple emotion tasks into a unified training dataset with explicit reasoning trajectories.
2. **Emo-Chord**: Performs an off-policy cold start, then introduces expert trajectory data from the same task family as an auxiliary target during GRPO, and performs credit assignment through a unified multi-task reward system, ultimately unlocking the reasoning potential of compact models.

## Resources

| Resource     | URL                                                                                | Purpose                                      |
| ------------ | ---------------------------------------------------------------------------------- | -------------------------------------------- |
| OneEmo-Base  | [Hugging Face Model](https://huggingface.co/Jiaha0Hu4ng/OneEmo-Base)              | Curriculum learning stage 1 weights          |
| OneEmo       | [Hugging Face Model](https://huggingface.co/Jiaha0Hu4ng/OneEmo)                   | Final Emo-Chord model weights                |
| EmoWorld-130K | [Hugging Face Dataset](https://huggingface.co/datasets/Jiaha0Hu4ng/EmoWorld-130K) | Unified emotion reasoning dataset            |
| Qwen3.5-Base | Please use the corresponding official base model repository                        | Initialization model for SFT and RL          |
| all-mpnet-base-v2 | [Hugging Face Model](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) | S-BERT model                            |
| Qwen2.5-7B-Instruct | [Hugging Face Model](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)        | RL thinking reward model                     |
| DeepSeek-V4-flash | -                                                                                  | Automated judge for interaction tasks        |
| MiMo-v2.5-Pro    | -                                                                                  | Automated judge for interaction tasks        |
| GPT-4.1-mini     | -                                                                                  | Automated judge for interaction tasks        |

## Directory Structure

```text
.
├── assets/logo.png                 # Multimedia assets
├── config.py                       # Configuration for raw data, labels, videos, and model paths
├── dataset.py                      # Raw video data loading and sample matching
├── datas/
│   ├── sftnew/                     # SFT data, including assistant reasoning and answers
│   ├── rl/                         # RL data, including solution, perception, and other reward fields
│   └── eval/                       # Inference/evaluation data except MerUnibench
├── reward/
│   ├── reward_plugin.py            # format, process, answer rewards
│   └── emotion_wheel/              # OVMER label mapping resources
├── scripts/                        # Scripts for SFT, GRPO, Emo-Chord, rollout, and LoRA weight merging
├── environment.yml                 # Conda environment
└── requirments.txt                 # Pip dependency list
```

## Environment Setup

### Key Version Dependencies

We used the following environment for training and evaluation; we recommend using the same configuration.
For training, a Linux GPU environment compatible with the CUDA driver is recommended, preferably with the locked versions provided in this repository.

| Component     | Version                |
| ------------- | ---------------------- |
| Python        | 3.12.0                 |
| PyTorch       | 2.10.0+cu128           |
| torchvision   | 0.25.0+cu128           |
| CUDA Runtime  | 12.8 (PyTorch `cu128`) |
| Transformers  | 5.2.0                  |
| ms-swift      | 4.1.3                  |
| vLLM          | 0.19.0                 |
| DeepSpeed     | 0.19.0                 |
| TRL           | 0.29.1                 |
| flash-attn    | 2.8.3                  |

### Install with Conda

```bash
conda env create -f environment.yml
conda activate oneemo
```

If you do not want to use the full Conda lock file directly, you can create a base environment with the following commands and then install the dependencies:

```bash
conda create -n oneemo python=3.12 -y
conda activate oneemo
pip install -r requirments.txt
```

## Model and Data Download

### OneEmo-Base (Curriculum stage1) Weights (if you want to skip SFT stage)

```bash
mkdir -p ckpts
hf download Jiaha0Hu4ng/OneEmo-Base \
  --local-dir ckpts/OneEmo-Base
```

### EmoWorld-130K

```bash
mkdir -p datas
hf download Jiaha0Hu4ng/EmoWorld-130K \
  --repo-type dataset \
  --local-dir datas
```

The training commands in `scripts/` directly read the in-repository prepared `datas/sftnew/*.json` and `datas/rl/*.json`. Therefore, downloading `EmoWorld-130K` alone will not automatically change the existing training data; to use the downloaded data, you need to organize it into the multimodal message format that `ms-swift` can read, and update the `--dataset` or `--chord_sft_dataset` argument in the corresponding scripts to the new JSON path.

### Qwen3.5-4B

The training scripts use a placeholder path `/path/to/your/OneEmo/ckpts/Qwen3.5-4B` for the Qwen3.5-4B. After downloading Qwen3.5-4B, please save it to a stable local directory, for example:

Then update the `--model` or `MODEL_PATH` placeholder in the following locations:

- `MODEL_PATH` in `config.py`, used as the default model path by `inference_swift.py` and `inference_vllm.py`;
- `scripts/train_sft.sh`;
- `scripts/train_sft_phase1.sh`;
- the base model path in `scripts/trans2Vllm_sft.sh`;
- any other script paths that still contain the `/path/to/your/resource/` placeholder.

You can use the following command to check for unreplaced placeholder paths:

```bash
rg -n "/path/to/your/OneEmo/ckpts/Qwen3.5-4B" config.py scripts datas
```

## Raw Video and Data Path Configuration

The repository centralizes raw data paths in `config.py`. After downloading or organizing the raw data, you need to check at least the following four categories of configuration:

```python
DATA_DIR = {
    "MIntRec": "/path/to/your/resource/MIntRec",
    "MIntRec2": "/path/to/your/resource/MIntRec2",
    # other datasets...
}

PATH_TO_RAW_VIDEO = {
    "MIntRec": os.path.join(DATA_DIR["MIntRec"], "video"),
    "MIntRec2": os.path.join(DATA_DIR["MIntRec2"], "video"),
}

PATH_TO_LABEL = {
    "MIntRec": os.path.join(DATA_DIR["MIntRec"], "label.npz"),
}

PATH_TO_TRANSCRIPTIONS = {
    "MIntRec": os.path.join(DATA_DIR["MIntRec"], "transcription-engchi-polish.csv"),
}
```

The actual fields and directory names must match your raw data directory. `dataset.py` uses these mappings to locate videos, labels, and transcriptions; `TESTSET_JSON` is used to read samples directly from evaluation JSON files. You should also check the following in `config.py`:

- `DATA_DIR`: root directory for each dataset;
- `PATH_TO_RAW_VIDEO`: video directory;
- `PATH_TO_LABEL`: label files;
- `PATH_TO_TRANSCRIPTIONS`: subtitle or transcription files;
- `TESTSET_JSON`: path to the evaluation JSON files.

Note: The `videos` field in `datas/sftnew/*.json` and `datas/rl/*.json` stores absolute paths to the original videos. Modifying `config.py` alone will not update these already generated JSON files; therefore, after downloading the data JSON files, you need to manually change the video paths inside them to the paths where you store the video data, or batch-update the `videos` paths in the JSON files. RL data should also preserve fields such as `task_type`, `solution`, `perception`, and `reasoning_ref`, otherwise the corresponding reward functions will not work properly.

## Curriculum Learning SFT

### Phase 1: Multi-task Cold Start

Phase 1 uses `scripts/train_sft_phase1.sh` and covers task families such as MSA, MER, MIR, MHD, MSD, and ERG. The default configuration uses LoRA, bfloat16, two GPUs, and DeepSpeed ZeRO-2:

```bash
conda activate oneemo
bash scripts/train_sft_phase1.sh
```

The default output directory is:

```text
ckpts/training_ckpts/sft_oneemo_p1/
```

The `CUDA_VISIBLE_DEVICES=0,1`, `NPROC_PER_NODE=2`, base model path, and data sampling size in the script can be adjusted according to your GPU memory and machine configuration.

### Export the Phase 1 Model

Phase 1 produces a LoRA checkpoint. Before running the export, modify the three paths in `scripts/trans2Vllm_sft.sh`:

```bash
--model <local path to Qwen3.5-4B>
--adapters <path to the Phase 1 checkpoint>
--output_dir <path to the exported full model>
```

Then run:

```bash
bash scripts/trans2Vllm_sft.sh
```

The exported directory must match the `--model` in the Phase 2 script. Currently `train_sft_phase2.sh` defaults to:

```text
ckpts/inference_ckpts/sft/oneemo_p1_e5_20260618
```

### Phase 2: Curriculum Learning Extension

Phase 2 continues training on the model exported from Phase 1, and adds more data and the ESC task:

```bash
bash scripts/train_sft_phase2.sh
```

The default output directory is:

```text
ckpts/training_ckpts/sft_oneemo_p2/
```

If you do not use the default directory, you need to update `--model` in `scripts/train_sft_phase2.sh` first. Phase 2 also uses LoRA; before subsequent inference or RL, you typically need to export the selected checkpoint as a merged full model.

## Emo-Chord Training

### Strategy Description

During training, this method reads a cyclic SFT data stream from `--chord_sft_dataset` and dynamically combines the SFT loss with the GRPO loss:

```text
L_CHORD = (1 - mu) * L_GRPO + mu * L_SFT
```

Here `mu` first warms up from 0 to `chord_mu_peak`, then decays via cosine annealing to `chord_mu_valley`; `chord_enable_phi_function` can enable token-level phi weights. This way, Emo-Chord first cold-starts from the model obtained in curriculum learning stage 1, and then preserves the SFT objective as a constraint during multi-task RL.

### Recommended Workflow: RL Training Directly After Phase 1 Cold Start

This is the workflow corresponding to `scripts/train_sft_then_chord.sh`:

```text
Qwen3.5-4B
      |
      v
SFT Stage 1 cold start
      |
      v
Export Stage 1 full model
      |
      v
vLLM rollout service + Emo-Chord
```

1. First complete Phase 1 SFT and model export above.
2. Modify `--model` in `scripts/rollout_grpo.sh` so that it points to the full model exported in Phase 1, then start the rollout service:
   ```bash
   CUDA_VISIBLE_DEVICES=1 bash ./scripts/rollout_grpo.sh
   ```
   The default service address is `127.0.0.1:8000`; the model path must be modified directly in the script.
3. Start the LLM-as-a-Judge service:
   ```bash
   bash ./scripts/start_judge.sh
   ```
   The default service address is `127.0.0.1:15555`, and the judge model is Qwen2.5-7B-Instruct.
4. Start the Emo-Chord training:
   ```bash
   bash ./scripts/train_sft_then_chord.sh
   ```

`train_sft_then_chord.sh` is already configured with `--chord_sft_dataset`, `--chord_mu_warmup_steps`, `--chord_mu_decay_steps`, `--chord_mu_peak`, `--chord_mu_valley`, and `--chord_enable_phi_function true`, and loads `reward/reward_plugin.py`.

### Run CHORD After Full Curriculum Learning

If you prefer to first complete Phase 1 and Phase 2, and then perform RL, you can use `scripts/train_grpo_chord.sh`:

1. Complete Phase 1 and Phase 2, and export the Phase 2 full model.
2. Modify `--model` in `scripts/rollout_grpo.sh` to the Phase 2 full model, and keep the rollout service running.
3. Write the Phase 2 model path and output directory into the CHORD script:
   ```bash
   bash ./scripts/train_grpo_chord.sh
   ```

This script uses the RL data directory `datas/rl/` by default, and uses `datas/sftnew/` as `--chord_sft_dataset`.

### Reward Functions and Dependencies

`reward/reward_plugin.py` registers three types of rewards:

- `format`: checks whether the output contains a single `<think>...</think>` reasoning block;
- `process`: evaluates factual consistency and reasoning-answer consistency via a judge service compatible with the OpenAI Chat Completions API; ERG/ESC uses the corresponding interaction evaluation dimensions;
- `answer`: computes label matching for classification tasks, multi-label WAF for OVMER, and text similarity using a local Sentence-BERT for ERG/ESC, combined with reasoning/answer length gating.

ERG/ESC rewards require a local Sentence-BERT model. When the repository already contains `ckpts/all-mpnet-base-v2`, set:

```bash
export VIDEMO_SENTENCE_BERT_MODEL=/path/to/your/OneEmo/ckpts/all-mpnet-base-v2
```

If you use the `process` reward in `train_sft_then_chord.sh`, you need to prepare a judge service and set the actual service address:

```bash
export VIDEMO_JUDGE_BASE_URL=http://127.0.0.1:15555/v1
export VIDEMO_JUDGE_MODEL=Qwen2.5-7B-Instruct
```

The default judge address in the code is an internal address of the current development environment and should not be treated as a universal configuration. `train_grpo_chord.sh` uses `format answer` by default and does not enable `process`; the actual reward combination is determined by `--reward_funcs` and `--reward_weights` in the script.

## Model Export and Inference

### LoRA Merge Export

SFT checkpoint export:

```bash
bash scripts/trans2Vllm_sft.sh
```

GRPO/CHORD checkpoint export:

```bash
bash scripts/trans2Vllm_grpo.sh
```

The base model, adapter checkpoint, and output directory in both scripts contain local absolute paths and need to be modified accordingly when reproducing.

### Python Inference Entry Points

`inference_swift.py` uses the TransformersEngine, while `inference_vllm.py` uses the vLLM/OpenAI-compatible interface. Both read raw video, label, and transcription configuration via `config.py` and `dataset.py`.

```bash
# Start the vLLM model first, then run inference
python inference_vllm.py --datasets merunibench --model_path {path} --verbose

# Directly use swift infer for inference
CUDA_VISIBLE_DEVICES=0 python inference_swift.py \
  --datasets merunibench \
  --model_path {path} \
  --verbose \
  --think true \
  --attn_impl sdpa
```

`--datasets` accepts dataset groups defined in `config.py`, such as `merunibench`, `mir`, `msd`, `mhd`, `erg`, and `esc`, or you can pass a single dataset name directly.


## Notes

- The training scripts default to a two-GPU SFT configuration and a single-GPU rollout/RL configuration; when GPU memory is insufficient, you need to adjust batch size, number of video frames, gradient accumulation, and the DeepSpeed configuration together.
- Multimodal training depends on environment variables such as `IMAGE_MAX_TOKEN_NUM=1024`, `VIDEO_MAX_TOKEN_NUM=128`, and `FPS_MAX_FRAMES=16`; these values can be tuned according to your GPU memory.
- The licenses of the original videos and each dataset are determined by their original publishers. Please follow the corresponding dataset licenses and terms of use before using the data.

## Citation

<!-- TODO: Add the BibTeX entry for the OneEmo paper here. -->

```bibtex
@article{oneemo,
  title     = {OneEmo: A Unified Multimodal Reasoning Model for Emotion Perception, Understanding, and Interaction},
  author    = {TODO},
  year      = {2026},
  url       = {https://arxiv.org/pdf/2603.02123}
}
```
