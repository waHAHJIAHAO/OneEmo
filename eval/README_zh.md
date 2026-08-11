# OneEmo 评估脚本说明（中文版）

本目录包含两个独立评估脚本，用于对 OneEmo 等模型进行 **感知与理解能力** 和 **交互质量（ERG/ESC）** 的自动化评估：

| 脚本 | 评估内容 |
|------|----------|
| `eval_perception_understanding.py` | 多模态情感感知/理解 benchmark（MER、MSA、MIR、MHU/MSU） |
| `eval_interaction.py` | 共情响应生成（ERG）与情感支持对话（ESC）质量评估 |

> 提醒：请按各节说明显式传参——`eval_perception_understanding.py` 的两个参数虽有默认值但建议显式指定，`eval_interaction.py` 的参数全部必填。

---

## 1. 环境与依赖

依赖已写入仓库根目录 `requirments.txt`，评估脚本额外用到的关键包：

- `openai`（LLM judge 调用，兼容 OpenAI 官方 API）
- `statsmodels`、`krippendorff`（评估一致性指标）
- `numpy`、`scipy`、`scikit-learn`、`pandas`
- `vllm`、`transformers`（`eval_perception_understanding.py` 在离散/维度/开集标签任务中需要，用于 openset 标签抽取）

```bash
pip install -r ../requirments.txt
```

`eval_perception_understanding.py` 需要 GPU（vLLM 推理），`eval_interaction.py` 仅调用远程 LLM judge，不需要本地 GPU。

---

## 2. 评估 LLM（Judge）配置

`eval_interaction.py` 使用 **3 个 LLM judge** 对每个样本打分，最后用中位数聚合。Judge 的模型名 / 接口地址 / API Key **全部从 `eval/.env` 读取，不硬编码在代码中**。

### 2.1 创建 `eval/.env`

```bash
# OpenAI（官方接口）
OpenAI_MODEL_NAME=gpt-5-mini
OpenAI_BASE_URL=https://api.openai.com/v1
OpenAI_API_KEY=sk-xxxx

# MIMO（官方接口，按需配置）
MIMO_EVAL_MODEL_NAME=mimo-v2.5
MIMO_BASE_URL=https://api.mimo.ai/v1
MIMO_API_KEY=sk-xxxx

# DeepSeek（官方接口，按需配置）
DeepSeek_MODEL_NAME=deepseek-chat
DeepSeek_BASE_URL=https://api.deepseek.com
DeepSeek_API_KEY=sk-xxxx
```

### 2.2 配置说明

- 文件必须命名为 `eval/.env`（脚本硬编码该路径：`PROJECT_ROOT/eval/.env`），不存在则直接报错。
- 三个 judge 的 `*_BASE_URL` / `*_API_KEY` / `*_MODEL_NAME` 都必须填齐，缺少任何一个 judge 会报 `Missing API configuration for judge`。
- **OpenAI**：base_url 不以 `/v1` 结尾时脚本会自动补 `/v1`，所以填 `https://api.openai.com/v1` 或 `https://api.openai.com` 均可。
- **DeepSeek**：base_url 不自动补 `/v1`，请填完整地址。
- **MIMO**：与 OpenAI 相同，自动补 `/v1`。
- 脚本只读取本文件中的键，未读到的 judge 相关键不会生效。

---

## 3. 任务 1：感知与理解能力评估（`eval_perception_understanding.py`）

该脚本读取模型在各类多模态情感数据集上的推理结果（npz），在**不打分模型、只看结果**（score-only）的前提下计算指标，并输出每轮（epoch）的最佳分数。

### 3.1 支持的 benchmark 与数据集


| `--datasets` 取值       | 数据集                                                                  | 任务               |
| --------------------- | -------------------------------------------------------------------- | ---------------- |
| `merunibench`（默认组）    | MER2023、MER2024、MELD、IEMOCAPFour、CMUMOSI、CMUMOSEI、SIMS、SIMSv2、OVMERD | MSA、B-MER、OV-MER |
| `intent`              | MIntRec、MIntRec2                                                     | MIR（多模态意图识别）     |
| `mustard_urfunny`（默认） | Mustard、URFunny                                                      | MSU、MHU（讽刺/幽默检测） |
| 单个数据集名                | 如 `mustard`、`urfunny`、`mintrec`、`meld`、`avamerg` 等                   | 仅该数据集            |


### 3.2 参数说明（重要）

```text
--modelname  待评估模型在 output/results-<dataset>/ 下的目录名（默认 rl_v1，务必显式指定）
--datasets   benchmark 集合：merunibench | intent | mustard_urfunny，或单个数据集名
```

- 两个参数都有默认值，但**强烈建议显式传参**，避免误用默认的 `rl_v1` / `mustard_urfunny`。
- 按数据集分组用英文逗号分隔也能识别（`--datasets a,b` 会按名字逐个处理）。

### 3.3 前置数据

每个数据集的推理结果需放在对应目录下，目录名必须含 `results-`：

```
output/results-<dataset>/<modelname>/checkpoint_<step>.npz
```

- npz 中需包含 `name2reason`（样本名 → 模型输出文本）。
- 脚本会自动挑目录中**最新的** checkpoint 根（若精确目录不存在，会按时间戳后缀检索最近的目录），并按文件名中的数字轮次依次评估每个 checkpoint，输出各轮分数。
- 中间文件 `*-openset.npz`、`*-sentiment.npz` 会被自动跳过。
- 数据集真值（`name2gt`）、音频/视频路径等由根目录 `config.py` 提供；脚本启动时会为评估动态补全 `EMOTION_WHEEL_ROOT`、`PATH_TO_RAW_AUDIO`、`PATH_TO_RAW_VIDEO['Mustard']`、`PATH_TO_LABEL` 等配置，无需修改根目录 config。

### 3.4 运行命令

```bash
# 在仓库根目录（OneEmo/）下运行；vLLM 需要 GPU
CUDA_VISIBLE_DEVICES=1 python eval/eval_perception_understanding.py \
    --modelname {待评估模型名称} --datasets mustard_urfunny

# 多模态情感识别 benchmark
CUDA_VISIBLE_DEVICES=1 python eval/eval_perception_understanding.py \
    --modelname {待评估模型名称} --datasets merunibench

# 意图识别 benchmark
CUDA_VISIBLE_DEVICES=1 python eval/eval_perception_understanding.py \
    --modelname {待评估模型名称} --datasets intent
```

> 注意：离散/维度/开集标签任务（MER、MSA、OV-MER 等）需要加载 `config.PATH_TO_LLM['Qwen25_7B']`（默认 `/public/home/lianzheng/hjh/AffectGPT/AffectGPT/models/Qwen2.5-7B-Instruct`）做 openset 标签抽取，请确保该路径在本机可用并配置到根目录 `config.py`。

### 3.5 输出

- 逐数据集打印最佳分数（含命中率 / F1 / ACC / Dist-1 / Dist-2 / 意图 ACC / WF1 / WP 等，取决于任务类型）及对应 epoch。
- 最后打印一行汇总表（各数据集分数 + 平均值），可直接用于横向对比。

---

## 4. 任务 2：交互质量评估 ERG / ESC（`eval_interaction.py`）

该脚本加载模型在 AvaMERG（ERG）与 OpenR1Psy（ESC）测试集上的推理结果，用 3 个 LLM judge 按 prompt 模板（`prompt.py` 中的 `ERG_EVAL_PROMPT` / `ESC_EVAL_PROMPT`）对回复打分，并计算一致性指标。

### 4.1 评估维度


| 任务          | 维度                                        |
| ----------- | ----------------------------------------- |
| ERG（共情响应生成） | Empathy、Coherence、Informativeness（1–5 整数） |
| ESC（情感支持对话） | Empathy、Skill、Overall（1–5 整数）             |


### 4.2 参数说明（重要）

```text
--task         erg | esc | all（all 表示两个任务都跑）
--provider     all | openai | mimo | deepseek   （限定 judge 提供方）
--judge_llm    all | openai | mimo | deepseek   （限定使用的 judge）
--model        {待评估模型名称}（必须与 results-<dataset>/ 下的目录名一致）
--debug        仅跑 2 个样本、不写结果文件（调试用，可选）
```

- `--task`、`--provider`、`--judge_llm`、`--model` 均为必填，**没有任何默认值，必须显式传参**。
- `--provider all` + `--judge_llm all` 会使用全部 3 个 judge（openai / mimo / deepseek），评分以**中位数**聚合，并计算一致性指标。
- 若 judge 数量不是 3（如只选 1 个 judge），脚本进入 **diagnostic 模式**：只输出各 judge 的原始评分，不聚合、不计算 alpha 等指标。
- `--provider` 与 `--judge_llm` 需匹配：例如 `--provider openai --judge_llm openai`，互不匹配会报错。

### 4.3 前置数据

- ERG 测试集：`datas/eval/testset_avamerg.json`
- ESC 测试集：`datas/eval/test_openr1psy.json`
- 模型推理结果（npz，含 `name2reason`）：

```
# ERG
/public/home/lianzheng/hjh/AffectGPT/AffectGPT/output/results-avamerg/<modelname>/checkpoint_*.npz
# ESC
/public/home/lianzheng/hjh/AffectGPT/AffectGPT/output/results-openr1psy/<modelname>/checkpoint_*.npz
```

- 脚本会从模型目录中选取**最新**的 checkpoint（按 `checkpoint_<step>` 的数字取最大）。
- npz 中的 `name2reason` 条数必须与 json 测试集条数一致，否则报错。
- 结果根目录 `RESULTS_ROOT` 默认是服务器路径（见脚本头部常量），如在本机评估请自行修改该常量。

### 4.4 运行命令

```bash
# 单个任务
python eval/eval_interaction.py --task erg \
    --provider all --judge_llm all --model {待评估模型名称}

python eval/eval_interaction.py --task esc \
    --provider all --judge_llm all --model {待评估模型名称}

# 两个任务一起跑
python eval/eval_interaction.py --task all \
    --provider all --judge_llm all --model {待评估模型名称}

# 只用一个 judge（diagnostic 模式，用于调试 prompt/解析）
python eval/eval_interaction.py --task esc \
    --provider openai --judge_llm openai --model {待评估模型名称}

# 快速调试（2 个样本，不写文件）
python eval/eval_interaction.py --task esc \
    --provider all --judge_llm all --model {待评估模型名称} --debug
```

> 本脚本不占用 GPU，只需要能访问 judge LLM 的 API 即可。

### 4.5 输出与断点续跑

输出目录：`eval/output/quality_eval/<task>/<model>/<checkpoint_stem>/`


| 文件                    | 内容                                               |
| --------------------- | ------------------------------------------------ |
| `sample_scores.jsonl` | 逐样本的原始 judge 输出、各维度分数、聚合分数                       |
| `summary.json`        | 汇总：各维度均值、Krippendorff's α、Randolph's κ、完全/两两一致率等 |
| `progress.json`       | 运行进度信息（断点续跑状态）                                   |


- 每完成 5 个样本自动落盘一次；中途中断后重新运行**相同参数**即可从上次进度继续（基于已完成的样本键去重），不会重复扣费。
- 结束后自动删除 `progress.json`，`sample_scores.jsonl` 与 `summary.json` 保留。
- `--debug` 模式不写任何结果文件。

---

## 5. 常见问题

- `**Missing env file: eval/.env`** → 先按第 2 节创建 `eval/.env`。
- `**Missing API configuration for judge: ...`** → `.env` 中对应 judge 的 `*_BASE_URL` / `*_API_KEY` / `*_MODEL_NAME` 未填全。
- `**Model directory does not exist`** → `--model` 名称与 `results-<dataset>/` 下目录名不一致，或 `RESULTS_ROOT` 路径不对。
- `**Sample count mismatch**` → npz 的 `name2reason` 数量与 json 测试集条数不一致，请检查推理输出是否完整。
- `**Unsupported judge_llm` / `is not compatible with provider**` → `--judge_llm` 取值需是 `all/openai/mimo/deepseek`，且与 `--provider` 匹配。
- **离散/维度任务无 GPU 报错** → 检查 vLLM 环境与 `config.PATH_TO_LLM['Qwen25_7B']` 路径。

