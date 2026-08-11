# OneEmo Evaluation Scripts (English)

This directory contains two independent evaluation scripts for automated assessment of OneEmo-style models:

| Script | What it evaluates |
|--------|-------------------|
| `eval_perception_understanding.py` | Multimodal emotion perception/understanding benchmarks (MER, MSA, MIR, MHU/MSU) |
| `eval_interaction.py` | Empathetic Response Generation (ERG) and Emotional Support Conversation (ESC) quality |

> Note: pass arguments explicitly as described in each section below — the two arguments of `eval_perception_understanding.py` have defaults but you should set them explicitly, and all arguments of `eval_interaction.py` are required.

---

## 1. Environment & Dependencies

Dependencies are listed in `requirments.txt` at the repository root. Key packages used by the eval scripts:

- `openai` (LLM judge calls; OpenAI-compatible API)
- `statsmodels`, `krippendorff` (inter-rater agreement metrics)
- `numpy`, `scipy`, `scikit-learn`, `pandas`
- `vllm`, `transformers` (required by `eval_perception_understanding.py` for openset label extraction on discrete/dimension/open-vocabulary tasks)

```bash
pip install -r ../requirments.txt
```

`eval_perception_understanding.py` requires a GPU (vLLM inference). `eval_interaction.py` only calls remote LLM judges and needs no local GPU.

---

## 2. Judge LLM Configuration

`eval_interaction.py` scores every sample with **3 LLM judges** and aggregates with the median. Judge model names / base URLs / API keys are **all read from `eval/.env` — nothing is hardcoded in the source**.

### 2.1 Create `eval/.env`

```bash
# OpenAI (official API)
OpenAI_MODEL_NAME=gpt-5-mini
OpenAI_BASE_URL=https://api.openai.com/v1
OpenAI_API_KEY=sk-xxxx

# MIMO (official API, configure as needed)
MIMO_EVAL_MODEL_NAME=mimo-v2.5
MIMO_BASE_URL=https://api.mimo.ai/v1
MIMO_API_KEY=sk-xxxx

# DeepSeek (official API, configure as needed)
DeepSeek_MODEL_NAME=deepseek-chat
DeepSeek_BASE_URL=https://api.deepseek.com
DeepSeek_API_KEY=sk-xxxx
```

### 2.2 Notes

- The file **must** be named `eval/.env` (hardcoded path: `PROJECT_ROOT/eval/.env`); the script raises an error if it is missing.
- All `*_BASE_URL` / `*_API_KEY` / `*_MODEL_NAME` keys for the three judges must be complete; a missing judge raises `Missing API configuration for judge`.
- **OpenAI**: `/v1` is appended automatically when missing, so `https://api.openai.com/v1` and `https://api.openai.com` both work.
- **DeepSeek**: `/v1` is NOT appended automatically — provide the full base URL.
- **MIMO**: same as OpenAI, `/v1` is appended automatically.
- Only keys present in this file are read; any leftover `Jianyi_*` keys from older versions are ignored.

---

## 3. Task 1: Perception & Understanding (`eval_perception_understanding.py`)

This score-only script reads model inference results (npz files) for various multimodal emotion datasets, computes metrics without re-running inference, and reports the best score across epochs.

### 3.1 Supported benchmarks & datasets

| `--datasets` value | Datasets | Task |
|--------------------|----------|------|
| `merunibench` | MER2023, MER2024, MELD, IEMOCAPFour, CMUMOSI, CMUMOSEI, SIMS, SIMSv2, OVMERD | MSA, B-MER, OV-MER |
| `intent` | MIntRec, MIntRec2 | MIR (multimodal intent recognition) |
| `mustard_urfunny` (default) | Mustard, URFunny | MSU, MHU (sarcasm/humor detection) |
| single dataset name | e.g. `mustard`, `urfunny`, `mintrec`, `meld`, `avamerg` | that dataset only |

### 3.2 Arguments (important)

```text
--modelname  Model directory name under output/results-<dataset>/ (default: rl_v1 — pass it explicitly)
--datasets   Benchmark collection: merunibench | intent | mustard_urfunny, or a single dataset name
```

- Both arguments have defaults, but you should **always pass them explicitly** to avoid accidentally using `rl_v1` / `mustard_urfunny`.
- A comma-separated list of dataset names is also accepted (`--datasets a,b`).

### 3.3 Input data

Inference results per dataset must be placed under a directory whose name contains `results-`:

```
output/results-<dataset>/<modelname>/checkpoint_<step>.npz
```

- Each npz must contain a `name2reason` dict (sample name → model output text).
- The script scans every checkpoint under the model directory (intermediate `*-openset.npz` / `*-sentiment.npz` files are skipped), evaluates each epoch identified by the number in the filename, and reports per-epoch and best scores.
- If the exact model directory does not exist, the script searches for the most recent directory by timestamp suffix.
- Ground-truth labels, audio/video paths, etc., come from `config.py` at the repo root. The script patches evaluation-only configs at runtime (`EMOTION_WHEEL_ROOT`, `PATH_TO_RAW_AUDIO`, `PATH_TO_RAW_VIDEO['Mustard']`, `PATH_TO_LABEL`, ...), so the root config does not need to be modified.

### 3.4 Usage

```bash
# Run from the repository root (OneEmo/); vLLM needs a GPU
CUDA_VISIBLE_DEVICES=1 python eval/eval_perception_understanding.py \
    --modelname {model_name} --datasets mustard_urfunny

# Multimodal emotion recognition benchmark
CUDA_VISIBLE_DEVICES=1 python eval/eval_perception_understanding.py \
    --modelname {model_name} --datasets merunibench

# Intent recognition benchmark
CUDA_VISIBLE_DEVICES=1 python eval/eval_perception_understanding.py \
    --modelname {model_name} --datasets intent
```

> For discrete / dimension / open-vocabulary tasks (MER, MSA, OV-MER, ...), the script loads `config.PATH_TO_LLM['Qwen25_7B']` (default `/public/home/lianzheng/hjh/AffectGPT/AffectGPT/models/Qwen2.5-7B-Instruct`) to extract openset labels. Make sure that path is valid on your machine and configured in the root `config.py`.

### 3.5 Output

- Per-dataset best scores (hitrate / F1 / ACC / Dist-1 / Dist-2 / intent ACC / WF1 / WP, depending on task type) with the corresponding epoch.
- A final summary table with per-dataset scores plus the average, ready for cross-model comparison.

---

## 4. Task 2: Interaction Quality ERG / ESC (`eval_interaction.py`)

This script loads model responses on the AvaMERG (ERG) and OpenR1Psy (ESC) test sets, asks 3 LLM judges to score each response with the prompt templates in `prompt.py` (`ERG_EVAL_PROMPT` / `ESC_EVAL_PROMPT`), and computes inter-rater agreement.

### 4.1 Evaluation dimensions

| Task | Dimensions |
|------|------------|
| ERG (empathetic response generation) | Empathy, Coherence, Informativeness (integer 1–5) |
| ESC (emotional support conversation) | Empathy, Skill, Overall (integer 1–5) |

### 4.2 Arguments (important)

```text
--task         erg | esc | all   (all runs both tasks)
--provider     all | openai | mimo | deepseek  (restrict judge providers)
--judge_llm    all | openai | mimo | deepseek  (restrict judges to use)
--model        {model_name}  (must match the directory under results-<dataset>/)
--debug        run only 2 samples and write no files (optional)
```

- `--task`, `--provider`, `--judge_llm`, and `--model` are all **required — no defaults, you must pass them explicitly**.
- `--provider all --judge_llm all` uses all 3 judges (openai / mimo / deepseek); scores are aggregated with the **median** and agreement metrics are computed.
- If the judge count is not 3 (e.g. a single judge), the script enters **diagnostic mode**: raw per-judge scores are printed, but no aggregation or alpha metrics are computed.
- `--provider` and `--judge_llm` must match (e.g. `--provider openai --judge_llm openai`); mismatches raise an error.

### 4.3 Input data

- ERG test set: `datas/eval/testset_avamerg.json`
- ESC test set: `datas/eval/test_openr1psy.json`
- Model inference results (npz with `name2reason`):

```
# ERG
/public/home/lianzheng/hjh/AffectGPT/AffectGPT/output/results-avamerg/<modelname>/checkpoint_*.npz
# ESC
/public/home/lianzheng/hjh/AffectGPT/AffectGPT/output/results-openr1psy/<modelname>/checkpoint_*.npz
```

- The script picks the **latest** checkpoint (largest `checkpoint_<step>` number) in the model directory.
- The number of `name2reason` entries in the npz must match the number of samples in the JSON test set, otherwise an error is raised.
- `RESULTS_ROOT` defaults to a server path (see the constant at the top of the script); change it if you evaluate on another machine.

### 4.4 Usage

```bash
# Single task
python eval/eval_interaction.py --task erg \
    --provider all --judge_llm all --model {model_name}

python eval/eval_interaction.py --task esc \
    --provider all --judge_llm all --model {model_name}

# Both tasks
python eval/eval_interaction.py --task all \
    --provider all --judge_llm all --model {model_name}

# Single judge (diagnostic mode, for debugging prompts/parsing)
python eval/eval_interaction.py --task esc \
    --provider openai --judge_llm openai --model {model_name}

# Quick debug (2 samples, no files written)
python eval/eval_interaction.py --task esc \
    --provider all --judge_llm all --model {model_name} --debug
```

> This script needs no GPU — only API access to the judge LLMs.

### 4.5 Output & resuming

Output directory: `eval/output/quality_eval/<task>/<model>/<checkpoint_stem>/`

| File | Contents |
|------|----------|
| `sample_scores.jsonl` | Per-sample raw judge outputs, per-dimension scores, aggregated scores |
| `summary.json` | Summary: per-dimension means, Krippendorff's α, Randolph's κ, full/pairwise agreement |
| `progress.json` | Run progress (resume state) |

- Progress is flushed to disk every 5 completed samples. If the run is interrupted, **re-run with the same arguments** to resume from the last progress (already-scored samples are skipped, so no extra cost).
- On success, `progress.json` is deleted while `sample_scores.jsonl` and `summary.json` are kept.
- `--debug` writes no result files.

---

## 5. FAQ

- **`Missing env file: eval/.env`** → Create `eval/.env` as described in Section 2.
- **`Missing API configuration for judge: ...`** → Some `*_BASE_URL` / `*_API_KEY` / `*_MODEL_NAME` is missing in `.env`.
- **`Model directory does not exist`** → `--model` does not match a directory under `results-<dataset>/`, or `RESULTS_ROOT` is wrong.
- **`Sample count mismatch`** → The npz `name2reason` count differs from the JSON test set; check whether inference output is complete.
- **`Unsupported judge_llm` / `is not compatible with provider`** → `--judge_llm` must be `all/openai/mimo/deepseek` and must match `--provider`.
- **GPU error on discrete/dimension tasks** → Check the vLLM environment and the `config.PATH_TO_LLM['Qwen25_7B']` path.
