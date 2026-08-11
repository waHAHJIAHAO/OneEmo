<p align="center">
  <img src="assets/logo.png" width="180" alt="OneEmo logo">
</p>

<h1 align="center">OneEmo: A Unified Multimodal Reasoning Model for Emotion Perception, Understanding, and Interaction</h1>

<p align="center">
    <a href="https://arxiv.org/pdf/2608.06013">
    <img src='https://img.shields.io/badge/Paper-Arxiv-orange' alt='Paper PDF'></a>
    <a href="https://huggingface.co/Jiaha0Hu4ng/OneEmo">
    <img src='https://img.shields.io/badge/Model-HuggingFace-yellow' alt='Model'></a>
    <a href='https://huggingface.co/datasets/Jiaha0Hu4ng/EmoWorld-130K'">
    <img src='https://img.shields.io/badge/Dataset-HuggingFace-yellow' alt='Dataset'></a>
</p>


## 概览

OneEmo 面向视频情感智能，将情绪感知、情绪理解和情绪交互统一到同一个多模态推理模型中，包括了Sentiment analysis，basic emotion recognition，Open-vocabulary emotion recognition，intention recognition，humor & sarcasm understanding, empathic response generation and emotional support conversation八个情感计算核心任务. OneEmo与相似规模的模型对比，在这些情感任务上取得了SOTA结果。训练流程包含两个核心部分：

1. **EmoWorld-130K**：从专家模型中将多个情感任务中的专业知识蒸馏为带有显式推理轨迹的统一训练数据集。
2. **Emo-Chord**：先进行Off-Policy冷启动，随后在 GRPO 过程中引入同任务族的专家轨迹数据作为辅助目标，并通过统一的多任务奖励系统进行信用分配，最终解锁紧凑模型的推理潜能。

## 资源

| 资源            | 地址                                                                                | 用途              |
| ------------- | --------------------------------------------------------------------------------- | --------------- |
| OneEmo-Base   | [Hugging Face Model](https://huggingface.co/Jiaha0Hu4ng/OneEmo-Base)                    | 课程学习第一阶段权重      |
| OneEmo   | [Hugging Face Model](https://huggingface.co/Jiaha0Hu4ng/OneEmo)                    | Emo-Chord最终模型权重      |
| EmoWorld-130K | [Hugging Face Dataset](https://huggingface.co/datasets/Jiaha0Hu4ng/EmoWorld-130K) | 统一情感推理数据集       |
| Qwen3.5-Base  | 请使用对应的官方基础模型仓库                                                          | SFT 和 RL 的初始化模型 |
| all-mpnet-base-v2 | [Hugging Face Model](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) | S-BERT模型       |
| Qwen2.5-7B-Instruct | [Hugging Face Model](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | RL思考奖励模型       |
| DeepSeek-V4-flash | - | 交互任务自动化指标评委       |
| MiMo-v2.5-Pro | - | 交互任务自动化指标评委       |
| GPT-4.1-mini | - | 交互任务自动化指标评委       |

## 目录结构

```text
.
├── assets/logo.png                 # 多媒体资源
├── config.py                       # 原始数据、标签、视频和模型路径配置
├── dataset.py                      # 原始视频数据读取和样本匹配
├── datas/
│   ├── sftnew/                     # SFT 数据，包含 assistant 推理和答案
│   ├── rl/                         # RL 数据，包含 solution、perception 等奖励字段
│   └── eval/                       # 除了MerUnibench之外的推理/评测数据
├── reward/
│   ├── reward_plugin.py            # format、process、answer 奖励
│   └── emotion_wheel/              # OVMER 标签映射资源
├── scripts/                        # SFT、GRPO、Emo-Chord、rollout和合并LoRA权重脚本
├── eval/                           # 评估脚本
│   ├── eval_perception_understanding.py  # 感知/理解 benchmark 评估
│   ├── eval_interaction.py         # ERG/ESC 交互质量评估（LLM 评委）
│   ├── prompt.py                   # ERG/ESC 评委 prompt 模板
│   ├── requirement.py              # 评估依赖清单
│   ├── emotion_wheel/              # OVMER 标签映射资源（评估用）
│   ├── my_affectgpt/               # 评估工具包（数据集/指标构建）
│   └── toolkit/                    # 工具（文件读取、qwen、vllm 辅助）
├── environment.yml                 # Conda 环境
└── requirments.txt                 # Pip 依赖清单
```

## 环境配置

### 关键版本依赖

我们在训练和评估的时候使用了下面的环境，推荐你使用同样的配置。
训练时建议使用与 CUDA 驱动兼容的 Linux GPU 环境，并优先使用仓库中的锁定版本。

| 组件           | 版本                    |
| ------------ | --------------------- |
| Python       | 3.12.0                |
| PyTorch      | 2.10.0+cu128          |
| torchvision  | 0.25.0+cu128          |
| CUDA Runtime | 12.8（PyTorch `cu128`） |
| Transformers | 5.2.0                 |
| ms-swift     | 4.1.3                 |
| vLLM         | 0.19.0                |
| DeepSpeed    | 0.19.0                |
| TRL          | 0.29.1                |
| flash-attn   | 2.8.3                 |

### 使用 Conda 安装

```bash
conda env create -f environment.yml
conda activate oneemo
```

如果不直接使用完整的 Conda 锁定文件，可以使用下面的方式创建基础环境，再安装依赖：

```bash
conda create -n oneemo python=3.12 -y
conda activate oneemo
pip install -r requirments.txt
```

## 模型与数据下载

### OneEmo-Base (Curriculum stage1) 权重

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
  --local-dir datas/EmoWorld-130K
```

当前 `scripts/` 中的训练命令直接读取仓库内已经整理好的 `datas/sftnew/*.json` 和 `datas/rl/*.json`。因此，单独下载 `EmoWorld-130K` 不会自动改变现有训练数据；如果要使用下载后的数据，需要将其整理为当前 `ms-swift` 能读取的多模态消息格式，并把对应脚本中的 `--dataset` 或 `--chord_sft_dataset` 改为新的 JSON 路径。

### Qwen3.5-4B

训练脚本当前把 Qwen3.5 基础模型写成了占位路径 `/path/to/your/resource/Qwen3.5-4B`。下载 Qwen3.5-Base 后，请将它保存到稳定的本地目录，例如：

```bash
export QWEN35_BASE=/path/to/your/OneEmo/ckpts/Qwen3.5-Base
```

然后修改以下位置中的 `--model` 或 `MODEL_PATH` 占位路径：

- `config.py` 中的 `MODEL_PATH`，用于 `inference_swift.py` 和 `inference_vllm.py` 的默认模型路径；
- `scripts/train_sft.sh`；
- `scripts/train_sft_phase1.sh`；
- `scripts/trans2Vllm_sft.sh` 的基础模型路径；
- 其他仍包含 `/path/to/your/resource/` 占位路径的脚本。

可用下面的命令检查尚未替换的占位路径：

```bash
rg -n "/path/to/your/resource|Qwen3\.5" config.py scripts datas
```

## 原始视频和数据路径配置

仓库把原始数据路径集中放在 `config.py` 中。下载或整理 raw data 后，至少需要检查以下四类配置：

```python
DATA_DIR = {
    "MIntRec": "/path/to/your/resource/MIntRec",
    "MIntRec2": "/path/to/your/resource/MIntRec2",
    # 其他数据集...
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

实际字段和目录名必须以你的 raw data 目录为准。`dataset.py` 使用这些映射寻找视频、标签和转录文本；`TESTSET_JSON` 则用于直接从评测 JSON 读取样本。需要同时检查 `config.py` 中的：

- `DATA_DIR`：每个数据集的根目录；
- `PATH_TO_RAW_VIDEO`：视频目录；
- `PATH_TO_LABEL`：标签文件；
- `PATH_TO_TRANSCRIPTIONS`：字幕或转录文件；
- `TESTSET_JSON`：评测 JSON 的路径。

注意：`datas/sftnew/*.json` 和 `datas/rl/*.json` 中的 `videos` 字段保存了原始视频的绝对路径。只修改 `config.py` 不会自动修改这些已经生成的 JSON；因此你下载完数据json文件后，需要手动修改里面的视频数据的路径为你存储的视频数据的路径，或批量更新 JSON 中的 `videos` 路径。RL 数据还应保留 `task_type`、`solution`、`perception` 和 `reasoning_ref` 等字段，否则对应奖励函数无法正常工作。

## 课程学习 SFT

### Phase 1：多任务冷启动

Phase 1 使用 `scripts/train_sft_phase1.sh`，覆盖 MSA、MER、MIR、MHD、MSD 和 ERG 等任务族。默认配置为 LoRA、bfloat16、两张 GPU、DeepSpeed ZeRO-2：

```bash
conda activate oneemo
bash scripts/train_sft_phase1.sh
```

默认输出目录为：

```text
ckpts/training_ckpts/sft_oneemo_p1/
```

脚本中的 `CUDA_VISIBLE_DEVICES=0,1`、`NPROC_PER_NODE=2`、基础模型路径和数据采样量可以根据显存和机器配置调整。

### 导出 Phase 1 模型

Phase 1 产出的是 LoRA checkpoint。运行导出前，修改 `scripts/trans2Vllm_sft.sh` 中的三个路径：

```bash
--model <Qwen3.5-4B 本地路径>
--adapters <Phase 1 的 checkpoint 路径>
--output_dir <导出的完整模型路径>
```

然后执行：

```bash
bash scripts/trans2Vllm_sft.sh
```

导出的目录需要与 Phase 2 脚本中的 `--model` 保持一致。当前 `train_sft_phase2.sh` 默认指向：

```text
ckpts/inference_ckpts/sft/oneemo_p1_e5_20260618
```

### Phase 2：课程学习扩展

Phase 2 在 Phase 1 导出的模型上继续训练，并加入更多数据和 ESC 任务：

```bash
bash scripts/train_sft_phase2.sh
```

默认输出目录为：

```text
ckpts/training_ckpts/sft_oneemo_p2/
```

如果没有使用当前默认目录，需要先修改 `scripts/train_sft_phase2.sh` 的 `--model`。Phase 2 也会使用 LoRA；后续进行推理或 RL 前，通常需要先把选定 checkpoint 导出为合并后的完整模型。

## Emo-Chord 训练

### 策略说明

该方法的训练时从 `--chord_sft_dataset` 读取一个循环 SFT 数据流，将 SFT loss 与 GRPO loss 动态地组合：

```text
L_Emo-Chord = (1 - mu) * L_GRPO + mu * L_SFT
```

其中 `mu` 先从 0 warm up 到 `chord_mu_peak`，再经过余弦衰减到 `chord_mu_valley`；`chord_enable_phi_function` 可以启用 token 级的 phi 权重。这样，Emo-Chord 先使用课程学习阶段一得到的模型进行冷启动，再在多任务 RL 中保留 SFT 目标的约束。

### 推荐流程：Phase 1 冷启动后直接强化学习训练

这是 `scripts/train_sft_then_chord.sh` 对应的流程：

```text
Qwen3.5-4B
      |
      v
SFT Stage 1 冷启动
      |
      v
导出 Stage 1 完整模型
      |
      v
vLLM rollout 服务 + Emo-Chord
```

1. 先完成上面的 Phase 1 SFT 和模型导出。
2. 修改 `scripts/rollout_grpo.sh` 中的 `--model`，使它指向 Phase 1 导出的完整模型，然后启动 rollout 服务：
   ```bash
   CUDA_VISIBLE_DEVICES=1 bash ./scripts/rollout_grpo.sh
   ```
   默认服务地址为 `127.0.0.1:8000`, 模型路径需要直接修改脚本。
3. 启动LLM-as-a-Judge 服务：
   ```bash
   bash ./scripts/start_judge.sh
   ```
   默认服务地址为 `127.0.0.1:15555`, Judge模型为Qwen2.5-7B-Instruct。
4. 启动 Emo-Chord 训练：
   ```bash
   bash ./scripts/train_sft_then_chord.sh
   ```

`train_sft_then_chord.sh` 已经配置了 `--chord_sft_dataset`、`--chord_mu_warmup_steps`、`--chord_mu_decay_steps`、`--chord_mu_peak`、`--chord_mu_valley` 和 `--chord_enable_phi_function true`，并加载 `reward/reward_plugin.py`。

### 完整课程学习后进行 Emo-Chord

如果希望先完成 Phase 1 和 Phase 2，再进行 RL，可以使用 `scripts/train_grpo_chord.sh`：

1. 完成 Phase 1 和 Phase 2，并导出 Phase 2 完整模型。
2. 修改 `scripts/rollout_grpo.sh` 的 `--model` 为 Phase 2 完整模型，并保持 rollout 服务运行。
3. 将 Phase 2 模型和输出目录写到 chord 脚本中：
   ```bash
   bash ./scripts/train_grpo_chord.sh
   ```

该脚本默认使用 RL 数据目录 `datas/rl/`，并使用 `datas/sftnew/` 作为 `--chord_sft_dataset`。

### 奖励函数和依赖

`reward/reward_plugin.py` 注册了三类奖励：

- `format`：检查输出是否包含单一的 `<think>...</think>` 推理块；
- `process`：通过兼容 OpenAI Chat Completions 的 judge 服务评估事实一致性、推理与答案一致性，ERG/ESC 使用相应的交互评价维度；
- `answer`：对分类任务计算标签匹配，对 OVMER 计算多标签 WAF，对 ERG/ESC 使用本地 Sentence-BERT 计算文本相似度，并结合推理/答案长度门控。

ERG/ESC 奖励需要本地 Sentence-BERT 模型。仓库已有 `ckpts/all-mpnet-base-v2` 时，设置：

```bash
export VIDEMO_SENTENCE_BERT_MODEL=/path/to/your/OneEmo/ckpts/all-mpnet-base-v2
```

如果使用 `train_sft_then_chord.sh` 中的 `process` 奖励，需要准备 judge 服务，并按实际服务地址设置：

```bash
export VIDEMO_JUDGE_BASE_URL=http://127.0.0.1:15555/v1
export VIDEMO_JUDGE_MODEL=Qwen2.5-7B-Instruct
```

代码中的默认 judge 地址是当前开发环境的内网地址，不应视为通用配置。`train_grpo_chord.sh` 默认使用 `format answer`，不启用 `process`；具体奖励组合以脚本中的 `--reward_funcs` 和 `--reward_weights` 为准。

## 模型导出和推理

### LoRA 合并导出

SFT checkpoint 导出：

```bash
bash scripts/trans2Vllm_sft.sh
```

GRPO/CHORD checkpoint 导出：

```bash
bash scripts/trans2Vllm_grpo.sh
```

这两个脚本中的基础模型、adapter checkpoint 和输出目录均包含本机绝对路径，需要自行在复现的时候修改。

### Python 推理入口

`inference_swift.py` 使用 TransformersEngine，`inference_vllm.py` 使用 vLLM/OpenAI 兼容接口。两者都会通过 `config.py` 和 `dataset.py` 读取原始视频、标签及转录配置。

```bash
# 先启动后推理 vLLM 模型
python inference_vllm.py --datasets merunibench --model_path {path} --verbose

#直接使用swift infer推理
CUDA_VISIBLE_DEVICES=0 python inference_swift.py \
  --datasets merunibench \
  --model_path {path} \
  --verbose \
  --think true \
  --attn_impl sdpa
```

`--datasets` 可以传入 `config.py` 中定义的数据集集合，例如 `merunibench`、`mir`、`msd`、`mhd`、`erg` 和 `esc`，也可以直接传入单个数据集名称。


## 说明

- 训练脚本默认面向两张 GPU 的 SFT 和单 GPU 的 rollout/RL 配置；显存不足时需要同时调整 batch size、视频帧数、梯度累积和 DeepSpeed 配置。
- 多模态训练依赖 `IMAGE_MAX_TOKEN_NUM=1024`、`VIDEO_MAX_TOKEN_NUM=128` 和 `FPS_MAX_FRAMES=16` 等环境变量；这些值可按显存调整。
- 原始视频及各数据集的许可证由其原始发布方决定。使用数据前请遵循对应数据集的许可和使用条款。

## 引用

如果你使用了本代码库，或认为我们的工作有价值，欢迎给一个 star :star: 并引用以下论文：

```bibtex
@article{huang2026oneemo,
      title={OneEmo: A Unified Multimodal Reasoning Model for Emotion Perception, Understanding, and Interaction},
      author={Jiahao Huang and Zheng Lian and Jingyi Zhang and Zhide Chen and Xiaojiang Peng and Shaonan Wang},
      year={2026},
      journal={arXiv preprint arXiv:2608.06013},
}
```
```bibtex
@InProceedings{Huang_2026_CVPR,
    author    = {Huang, Jiahao and Lin, Fengyan and Yang, Xuechao and Feng, Chen and Zhu, Kexin and Yang, Xu and Chen, Zhide},
    title     = {Nano-EmoX: Unifying Multimodal Emotional Intelligence from Perception to Empathy},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {22986-22997}
}
```

## 致谢

OneEmo 的训练与评测流程建立在多个优秀的开源工作之上。我们在此特别感谢：

- [ms-swift](https://github.com/modelscope/ms-swift)：为本项目的 SFT、LoRA 合并导出与多模态推理提供了高效且易用的训练框架。
- [AffectGPT](https://github.com/zeroQiaoba/AffectGPT)：其数据集与评测工具链为本项目的情感理解基准构建（`eval/my_affectgpt/`）提供了重要参考与基础。

