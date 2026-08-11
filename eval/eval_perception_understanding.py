import os
import sys

# ---- path setup -------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))          # OneEmo/eval
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)                       # OneEmo
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)     # 让 `import config` 可用
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)      # 让 toolkit / my_affectgpt 可用

# ---- import & patch config --------------------------------------------------
import config

# 评估专用: Emotion Wheel 资源路径
if not hasattr(config, 'EMOTION_WHEEL_ROOT'):
    config.EMOTION_WHEEL_ROOT = os.path.join(CURRENT_DIR, 'emotion_wheel')

# 评估专用: openset 抽取所用的 LLM 模型路径
if not hasattr(config, 'PATH_TO_LLM'):
    config.PATH_TO_LLM = {
        'Qwen25_7B': '/path/to/your/OneEmo/ckpts/Qwen2.5-7B-Instruct',
    }

# 评估专用: 音频路径 (dataset 类初始化需要)
if not hasattr(config, 'PATH_TO_RAW_AUDIO'):
    _DATA_DIR = config.DATA_DIR
    config.PATH_TO_RAW_AUDIO = {
        'MER2025OV':      os.path.join(_DATA_DIR['MER2025OV'], 'audio'),
        'MERCaptionPlus': os.path.join(_DATA_DIR['MERCaptionPlus'], 'audio'),
        'OVMERD':         os.path.join(_DATA_DIR['OVMERD'], 'audio'),
        'MER2023':        os.path.join(_DATA_DIR['MER2023'], 'audio'),
        'IEMOCAPFour':    os.path.join(_DATA_DIR['IEMOCAPFour'], 'subaudio'),
        'CMUMOSI':        os.path.join(_DATA_DIR['CMUMOSI'], 'subaudio'),
        'CMUMOSEI':       os.path.join(_DATA_DIR['CMUMOSEI'], 'subaudio'),
        'SIMS':           os.path.join(_DATA_DIR['SIMS'], 'audio'),
        'MELD':           os.path.join(_DATA_DIR['MELD'], 'subaudio'),
        'SIMSv2':         os.path.join(_DATA_DIR['SIMSv2'], 'audio'),
        'MER2024':        os.path.join(_DATA_DIR['MER2024'], 'audio'),
        'AvaMERG':        os.path.join(_DATA_DIR['AvaMERG'], 'audio'),
        'MERRFine':       os.path.join(_DATA_DIR['MERRFine'], 'audio'),
        'MIntRec':        os.path.join(_DATA_DIR['MIntRec'], 'audio'),
        'MIntRec2':       os.path.join(_DATA_DIR['MIntRec2'], 'audio'),
    }

# 评估专用: Mustard 视频路径 (dataset 类初始化可能需要)
if 'Mustard' not in config.PATH_TO_RAW_VIDEO:
    config.PATH_TO_RAW_VIDEO['Mustard'] = os.path.join(config.DATA_DIR['Mustard'], 'videos')

# 评估专用: Mustard / URFunny 的二分类真值标签路径
# OneEmo 的 config.TESTSET_JSON 已包含这两个数据集, 复用之
if not hasattr(config, 'PATH_TO_LABEL'):
    config.PATH_TO_LABEL = {}
if 'Mustard' not in config.PATH_TO_LABEL and 'Mustard' in config.TESTSET_JSON:
    config.PATH_TO_LABEL['Mustard'] = config.TESTSET_JSON['Mustard']
if 'URFunny' not in config.PATH_TO_LABEL and 'URFunny' in config.TESTSET_JSON:
    config.PATH_TO_LABEL['URFunny'] = config.TESTSET_JSON['URFunny']

# ---- 以下为原始评估逻辑 (保持不变) -------------------------------------------
import re
import time
import copy
import tqdm
import glob
import json
import math
import scipy
import shutil
import random
import pickle
import argparse
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from toolkit.utils.read_files import *
from toolkit.utils.qwen import *
from toolkit.utils.functions import *

def load_name2reason_from_npz(npz_path):
    """从NPZ文件加载name2reason数据"""
    try:
        data = np.load(npz_path, allow_pickle=True)
        if 'name2reason' in data:
            return data['name2reason'].item()
        else:
            print(f"警告：NPZ文件中没有找到'name2reason'键: {npz_path}")
            return None
    except Exception as e:
        print(f"加载NPZ文件失败 {npz_path}: {e}")
        return None

def search_for_result_root(input_dir, inter_print=True):
    candidates = glob.glob(input_dir + '*')
    root_candidates = [root for root in candidates if os.path.isdir(root)]
    print(f"root_candidates is:{root_candidates}" )
    if len(root_candidates) == 0:
        if inter_print: print ('No file exists!')
        return ''


    # 找到最新的评估结果root
    maxcount = 0
    maxtimestampe = 0
    targetroot = ''
    for root in root_candidates:
        match = re.search(r'(\d{11})$', root)
        timestamp = int(match.group(1)) if match else 0
        if timestamp > maxtimestampe:
            maxtimestampe = timestamp
            targetroot = root

    if inter_print: print ('================================================')
    if inter_print: print ('Targetroot: ', targetroot)
    if inter_print: print ('Saved result files ', maxcount)
    # report last file info
    last_file = sorted(glob.glob(targetroot + '/checkpoint*'))[-1]
    file_stat = Path(last_file).stat()
    creation_time = file_stat.st_ctime
    if inter_print: print("Last result file creation time:", datetime.datetime.fromtimestamp(creation_time))
    if inter_print: print ('================================================')
    return targetroot


def func_read_datasetname(input_dir):
    supprot_datasets = list(config.DATA_DIR.keys())
    assert input_dir.find('/results-') != -1
    dataset = input_dir.split('/results-')[1].split('/')[0]
    for supprot_item in supprot_datasets:
        if supprot_item.lower() == dataset.lower():
            return supprot_item
    ValueError(f'cannot find suitable dataset for {input_dir}')


def get_dataset2cls(dataset):
    from my_affectgpt.datasets.builders.image_text_pair_builder import (
        MER2023_Dataset,
        MER2024_Dataset,
        MELD_Dataset,
        IEMOCAPFour_Dataset,
        CMUMOSI_Dataset,
        CMUMOSEI_Dataset,
        SIMS_Dataset,
        SIMSv2_Dataset,
        MER2025OV_Dataset,
        AvaMERG_Dataset,
        OVMERD_Dataset,
        MIntRec_Dataset,
        MIntRec2_Dataset,
    )

    if dataset == 'MER2023':     return MER2023_Dataset()
    if dataset == 'MER2024':     return MER2024_Dataset()
    if dataset == 'MELD':        return MELD_Dataset()
    if dataset == 'IEMOCAPFour': return IEMOCAPFour_Dataset()
    if dataset == 'CMUMOSI':     return CMUMOSI_Dataset()
    if dataset == 'CMUMOSEI':    return CMUMOSEI_Dataset()
    if dataset == 'SIMS':        return SIMS_Dataset()
    if dataset == 'SIMSv2':      return SIMSv2_Dataset()
    if dataset == 'MER2025OV':   return MER2025OV_Dataset()
    if dataset == 'AvaMERG':     return AvaMERG_Dataset()
    if dataset == 'OVMERD':      return OVMERD_Dataset()
    if dataset == 'MIntRec':      return MIntRec_Dataset()
    if dataset == 'MIntRec2':     return MIntRec2_Dataset()
    print ('dataset cls not provided!')
    return None


def get_discrete_or_dimension_flag(dataset):
    if dataset in ['MER2023', 'MER2024', 'MELD', 'IEMOCAPFour']:
        return 'discrete'
    elif dataset in ['CMUMOSI', 'CMUMOSEI', 'SIMS', 'SIMSv2']:
        return 'dimension'
    elif dataset in ['MER2025OV', 'OVMERD']:
        return 'ovlabel'
    elif dataset in ['AvaMERG']:
        return 'avamerg'
    elif dataset in ['MIntRec', 'MIntRec2']:
        return 'mintrec'
    elif dataset in ['Mustard', 'URFunny']:
        return 'binary_text'
    else:
        ValueError('unsupported dataset input')


def get_emo2idx_idx2emo(dataset_cls):
    emo2idx, idx2emo = {}, {}

    if hasattr(dataset_cls, 'get_emo2idx_idx2emo'):
        emo2idx, idx2emo = dataset_cls.get_emo2idx_idx2emo()
        # post process [不同数据集的标签表示有些许差异，进行统一化处理]
        if 'happy' in emo2idx: emo2idx['joy']   = emo2idx['happy']
        if 'anger' in emo2idx: emo2idx['angry'] = emo2idx['anger']
        if 'sad'   in emo2idx: emo2idx['sadness'] = emo2idx['sad']
        if 'joy'   in emo2idx: emo2idx['happy'] = emo2idx['joy']
        if 'angry' in emo2idx: emo2idx['anger'] = emo2idx['angry']
    return emo2idx, idx2emo

def func_read_batch_calling_model(modelname):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    model_path = config.PATH_TO_LLM[modelname]
    llm = LLM(model=model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
    return llm, tokenizer, sampling_params


## similarity score for: openset <-> discrete
def calculate_discrete_zeroshot(epoch_root, name2gt, llm, tokenizer, sampling_params, inter_print=True):
    from my_affectgpt.evaluation.ew_metric import hitrate_metric_calculation
    from my_affectgpt.evaluation.ew_metric import extract_openset_batchcalling
    from vllm import SamplingParams

    # epoch_root=(name2reason) => openset
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
    openset_npz = epoch_root[:-4]+'-openset.npz'
    if not os.path.exists(openset_npz):
        # 加载name2reason数据
        name2reason_data = load_name2reason_from_npz(epoch_root)
        if name2reason_data is None:
            print(f"无法从 {epoch_root} 加载name2reason数据")
            return 0, 0

        extract_openset_batchcalling(name2reason=name2reason_data, store_npz=openset_npz,
                                     llm=llm, tokenizer=tokenizer, sampling_params=sampling_params)
    # 计算 hitrate, mscore
    hitrate, mscore = hitrate_metric_calculation(name2gt=name2gt, openset_npz=openset_npz, inter_print=inter_print)
    return hitrate, mscore


def calculate_ov_zeroshot(epoch_root, name2gt, llm, tokenizer, sampling_params, inter_print=True):
    from my_affectgpt.evaluation.ew_metric import extract_openset_batchcalling
    from my_affectgpt.evaluation.wheel import wheel_metric_calculation

    # epoch_root=(name2reason) => openset
    openset_npz = epoch_root[:-4]+'-openset.npz'
    if not os.path.exists(openset_npz):
        # 加载name2reason数据
        name2reason_data = load_name2reason_from_npz(epoch_root)
        if name2reason_data is None:
            print(f"无法从 {epoch_root} 加载name2reason数据")
            return 0, 0

        extract_openset_batchcalling(name2reason=name2reason_data, store_npz=openset_npz,
                                     llm=llm, tokenizer=tokenizer, sampling_params=sampling_params)

    # 计算 EW-based metrics
    name2pred = {}
    filenames = np.load(openset_npz, allow_pickle=True)['filenames']
    fileitems = np.load(openset_npz, allow_pickle=True)['fileitems']
    for (name, item) in zip(filenames, fileitems):
        name2pred[name] = item
    fscore, precision, recall = wheel_metric_calculation(name2gt=name2gt, name2pred=name2pred, inter_print=inter_print)
    return fscore, precision, recall


## similarity score for: openset -> sentiment <-> sentiment
def calculate_dimension_zeroshot(epoch_root, name2gt, llm, tokenizer, sampling_params, inter_print=True):
    from my_affectgpt.evaluation.ew_metric import extract_openset_batchcalling, openset_to_sentiment_batchcalling

    # 1. 抽取 openset
    openset_npz = epoch_root[:-4]+'-openset.npz'
    if not os.path.exists(openset_npz):
        # 加载name2reason数据
        name2reason_data = load_name2reason_from_npz(epoch_root)
        if name2reason_data is None:
            print(f"无法从 {epoch_root} 加载name2reason数据")
            return 0, 0

        extract_openset_batchcalling(name2reason=name2reason_data, store_npz=openset_npz,
                                     llm=llm, tokenizer=tokenizer, sampling_params=sampling_params)

    # 2. 将 openset 转成 [positive, negative, neutral]
    sentiment_npz = openset_npz[:-4]+'-sentiment.npz'
    if not os.path.exists(sentiment_npz):
        openset_to_sentiment_batchcalling(openset_npz=openset_npz, store_npz=sentiment_npz,
                                          llm=llm, tokenizer=tokenizer, sampling_params=sampling_params)

    # 3. 计算 scores
    ## 3.0 openset 自然语言形式标签 -> float
    name2pred = {}
    filenames = np.load(sentiment_npz, allow_pickle=True)['filenames']
    fileitems = np.load(sentiment_npz, allow_pickle=True)['fileitems']
    for (name, item) in zip(filenames, fileitems):
        item_clean = item.strip().strip('"').strip("'")
        if item_clean == 'positive':
            name2pred[name] = 1
        elif item_clean == 'negative':
            name2pred[name] = -1
        elif item_clean == 'neutral':
            name2pred[name] = 0
        else: # 其他无法操作的标签
            if inter_print: print ('error sample:', name, item, '-> cleaned:', item_clean)
            name2pred[name] = 0
    ## 3.1 conversion
    val_labels, val_preds = [], []
    for name in name2gt:
        val_labels.append(name2gt[name])
        val_preds.append(name2pred[name])
    val_labels = np.array(val_labels)
    val_preds = np.array(val_preds)
    ## 3.2 metric calculation (name2gt, name2pred) -> scores
    non_zeros = np.array([i for i, e in enumerate(val_labels) if e != 0]) # remove 0, and remove mask
    accuracy = accuracy_score((val_labels[non_zeros] > 0), (val_preds[non_zeros] > 0))
    fscore = f1_score((val_labels[non_zeros] > 0), (val_preds[non_zeros] > 0), average='weighted')
    return fscore, accuracy


def calculate_avamerg_metrics(epoch_root, name2gt, inter_print=True):
    from my_affectgpt.evaluation.eval_merg_exp import evaluate_avamerg_metrics
    """
    计算AvaMERG数据集的评估指标：情感准确率(Acc)、Dist-1、Dist-2
    """
    # 读取推理结果
    name2reason = np.load(epoch_root, allow_pickle=True)['name2reason'].item()

    # 使用eval_merg_exp中的评估函数
    results = evaluate_avamerg_metrics(name2gt, name2reason)

    emotion_acc = results["Emotion_Accuracy"]
    hitrate = results["Hitrate"]
    dist1 = results["Dist-1"]
    dist2 = results["Dist-2"]

    if inter_print:
        print(f'Emotion Accuracy: {emotion_acc:.4f}')
        print(f'Hitrate: {hitrate:.4f}')
        print(f'Dist-1: {dist1:.4f}')
        print(f'Dist-2: {dist2:.4f}')

    return emotion_acc, hitrate, dist1, dist2


def calculate_mintrec_metrics(epoch_root, name2gt, inter_print=True):
    from my_affectgpt.evaluation.eval_merg_exp import evaluate_mintrec_metrics
    """
    计算MIntRec数据集的评估指标：意图准确率(ACC)、加权F1(Weighted F1)、加权精确率(Weighted Precision)
    """
    # 读取推理结果
    name2reason = np.load(epoch_root, allow_pickle=True)['name2reason'].item()

    # 使用eval_merg_exp中的评估函数
    results = evaluate_mintrec_metrics(name2gt, name2reason)

    intent_acc = results["Intent_Accuracy"]
    weighted_f1 = results["Weighted_F1"]
    weighted_precision = results["Weighted_Precision"]

    if inter_print:
        print(f'Intent Accuracy: {intent_acc:.4f}')
        print(f'Weighted F1: {weighted_f1:.4f}')
        print(f'Weighted Precision: {weighted_precision:.4f}')

    return intent_acc, weighted_f1, weighted_precision


def normalize_binary_answer(response_text):
    """抽取并清洗二分类答案，最终统一到 yes / no。"""
    if not response_text:
        return ""

    text = str(response_text)

    if "</think>" in text:
        text = text.split("</think>", 1)[1]

    if "<answer>" in text and "</answer>" in text:
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1)

    text = text.lower().replace("\n", " ").replace("\r", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    if "yes" in tokens:
        return "yes"
    if "no" in tokens:
        return "no"
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    return text


def load_binary_text_name2gt(dataset):
    """读取 Mustard / URFunny 的 ShareGPT 格式测试真值。"""
    ann_path = config.PATH_TO_LABEL[dataset]
    with open(ann_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    name2gt = {}
    for item in annotations:
        if "videos" not in item or len(item["videos"]) == 0:
            continue

        sample_name = os.path.splitext(os.path.basename(item["videos"][0]))[0]
        assistant_content = ""
        for message in item.get("messages", []):
            if message.get("role") == "assistant":
                assistant_content = message.get("content", "")
                break

        label = normalize_binary_answer(assistant_content)
        if label not in {"yes", "no"}:
            print(f"warning: invalid gt label for {dataset}/{sample_name}: {label}")
            continue
        name2gt[sample_name] = label
    return name2gt


def calculate_binary_text_metrics(epoch_root, name2gt, inter_print=True):
    """
    计算 Mustard / URFunny 二分类指标:
    Acc, WF1, WP, precision, F1, Recall
    """
    name2reason = np.load(epoch_root, allow_pickle=True)["name2reason"].item()

    y_true, y_pred = [], []
    missing_pred_names, invalid_pred_names = [], []
    label2idx = {"no": 0, "yes": 1}

    for name, gt_text in name2gt.items():
        if name not in name2reason:
            missing_pred_names.append(name)
            continue

        pred_text = normalize_binary_answer(name2reason[name])
        if pred_text not in label2idx:
            invalid_pred_names.append((name, pred_text))
            continue

        y_true.append(label2idx[gt_text])
        y_pred.append(label2idx[pred_text])

    if len(y_true) == 0:
        if inter_print:
            print(f"warning: no valid samples found in {epoch_root}")
        return 0, 0, 0, 0, 0, 0

    acc = accuracy_score(y_true, y_pred)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    precision = precision_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)

    acc = round(acc * 100, 2)
    weighted_f1 = round(weighted_f1 * 100, 2)
    weighted_precision = round(weighted_precision * 100, 2)
    precision = round(precision * 100, 2)
    f1 = round(f1 * 100, 2)
    recall = round(recall * 100, 2)

    if inter_print:
        if missing_pred_names:
            print(f"warning: missing predictions: {len(missing_pred_names)}")
        if invalid_pred_names:
            print(f"warning: invalid predictions: {len(invalid_pred_names)}")
        print(f'Acc: {acc}')
        print(f'WF1: {weighted_f1}')
        print(f'WP: {weighted_precision}')
        print(f'precision: {precision}')
        print(f'F1: {f1}')
        print(f'Recall: {recall}')

    return acc, weighted_f1, weighted_precision, precision, f1, recall


def main_zeroshot_scores(input_dir, debug=False, test_epochs='', inter_print=True):

    # ## 如果 input_dir 不存在的话，那么需要去检索最匹配的路径
    if not os.path.exists(input_dir):
        input_dir = search_for_result_root(input_dir, inter_print)
    if inter_print: print (f'process root: {input_dir}')

    # read dataset infos
    dataset = func_read_datasetname(input_dir)
    disordim_flag = get_discrete_or_dimension_flag(dataset)
    if inter_print: print (f'process dataset: {dataset} => {disordim_flag}')
    dataset_cls = None
    if disordim_flag == 'binary_text':
        name2gt = load_binary_text_name2gt(dataset)
    else:
        dataset_cls = get_dataset2cls(dataset)
        if dataset_cls is None:
            raise ValueError(f'Dataset class not found for {dataset}')
        name2gt = dataset_cls.get_test_name2gt()
    if inter_print: print (f'target sample number: {len(name2gt)}')

    # discrete: 自然语言形式标签；dimension: float score
    if disordim_flag == 'discrete':
        _, idx2emo = get_emo2idx_idx2emo(dataset_cls)
        # => update (name2gt)
        for name in name2gt:
            gt = name2gt[name]
            if not isinstance(gt, str):
                name2gt[name] = idx2emo[gt]

    # load model
    llm, tokenizer, sampling_params = None, None, None
    if debug == False and disordim_flag in ['discrete', 'dimension', 'ovlabel']:
        llm, tokenizer, sampling_params = func_read_batch_calling_model(modelname='Qwen25_7B')

    # main process
    whole_score1s, whole_score2s, whole_score3s = [], [], []
    whole_score4s, whole_score5s, whole_score6s = [], [], []
    epoch_numbers = []  # 记录每个epoch的轮次
    for epoch_root in sorted(glob.glob(input_dir + '/*.npz')):

        if epoch_root.find('openset') != -1 or epoch_root.find('sentiment') != -1:
            continue

        # =============== process for {epoch_root} ===============
        epochname = os.path.basename(epoch_root)
        # 提取数字轮次，文件名中未含数字则跳过（如 test.npz）
        match_epoch = re.search(r"(\d+)", epochname)
        if match_epoch is None:
            continue
        cur_epoch = int(match_epoch.group(1))
        if inter_print: print (epochname)
        # 0. 判断 epoch 是不是在 test_epochs 内，否是就跳过这部分
        if test_epochs != '':
            run_epochs = [int(item) for item in test_epochs.split(',')]
            if cur_epoch not in run_epochs:
                continue

        epoch_numbers.append(cur_epoch)

        # 1. score calculation
        if disordim_flag == 'discrete':
            hitrate, _ = calculate_discrete_zeroshot(epoch_root, name2gt, llm, tokenizer, sampling_params, inter_print)
            if inter_print: print(f'hitrate: {hitrate}')
            whole_score1s.append(hitrate)
            whole_score2s.append(0)
            whole_score3s.append(0)
            whole_score4s.append(0)
            whole_score5s.append(0)
            whole_score6s.append(0)

        elif disordim_flag == 'dimension':
            fscore, acc = calculate_dimension_zeroshot(epoch_root, name2gt, llm, tokenizer, sampling_params, inter_print)
            if inter_print: print(f'fscore: {fscore}, acc: {acc}')
            whole_score1s.append(fscore)
            whole_score2s.append(acc)
            whole_score3s.append(0)
            whole_score4s.append(0)
            whole_score5s.append(0)
            whole_score6s.append(0)

        elif disordim_flag == 'ovlabel':
            fscore, precision, recall = calculate_ov_zeroshot(epoch_root, name2gt, llm, tokenizer, sampling_params, inter_print)
            if inter_print: print(f'fscore: {fscore}, precision: {precision}, recall: {recall}')
            whole_score1s.append(fscore)
            whole_score2s.append(precision)
            whole_score3s.append(recall)
            whole_score4s.append(0)
            whole_score5s.append(0)
            whole_score6s.append(0)

        elif disordim_flag == 'avamerg':
            emotion_acc, hitrate, dist1, dist2 = calculate_avamerg_metrics(epoch_root, name2gt, inter_print)
            if inter_print: print(f'emotion_acc: {emotion_acc}, hitrate: {hitrate}, dist1: {dist1}, dist2: {dist2}')
            whole_score1s.append(emotion_acc)
            whole_score2s.append(hitrate)
            whole_score3s.append(dist1)
            whole_score4s.append(dist2)
            whole_score5s.append(0)
            whole_score6s.append(0)

        elif disordim_flag == 'mintrec':
            intent_acc, weighted_f1, weighted_precision = calculate_mintrec_metrics(epoch_root, name2gt, inter_print)
            if inter_print: print(f'intent_acc: {intent_acc}, weighted_f1: {weighted_f1}, weighted_precision: {weighted_precision}')
            whole_score1s.append(intent_acc)
            whole_score2s.append(weighted_f1)
            whole_score3s.append(weighted_precision)
            whole_score4s.append(0)
            whole_score5s.append(0)
            whole_score6s.append(0)

        elif disordim_flag == 'binary_text':
            acc, weighted_f1, weighted_precision, precision, f1, recall = calculate_binary_text_metrics(epoch_root, name2gt, inter_print)
            if inter_print:
                print(f'acc: {acc}, weighted_f1: {weighted_f1}, weighted_precision: {weighted_precision}, precision: {precision}, f1: {f1}, recall: {recall}')
            whole_score1s.append(acc)
            whole_score2s.append(weighted_f1)
            whole_score3s.append(weighted_precision)
            whole_score4s.append(precision)
            whole_score5s.append(f1)
            whole_score6s.append(recall)

        if inter_print: print ('=========================')

    if len(whole_score1s) == 0:
        if inter_print:
            print('No valid checkpoint npz found under', input_dir)
        return 0, 0, 0, 0, -1

    # whole_score1s => main metric
    best_index = np.argmax(whole_score1s)
    best_score1 = whole_score1s[best_index]
    best_score2 = whole_score2s[best_index]
    best_score3 = whole_score3s[best_index]
    best_score4 = whole_score4s[best_index]
    best_score5 = whole_score5s[best_index]
    best_score6 = whole_score6s[best_index]
    best_epoch = epoch_numbers[best_index]  # 获取最佳轮次

    if disordim_flag == 'discrete':
        if inter_print: print (f'{dataset}: best hitrate: %.4f; best mscore: %.4f (epoch: {best_epoch})' %(best_score1, best_score2))
    elif disordim_flag == 'dimension':
        if inter_print: print (f'{dataset}: best fscore: %.4f; best acc: %.4f (epoch: {best_epoch})' %(best_score1, best_score2))
    elif disordim_flag == 'ovlabel':
        if inter_print: print (f'{dataset}: best fscore: %.4f; best precision: %.4f; best recall: %.4f (epoch: {best_epoch})' %(best_score1, best_score2,best_score3))
    elif disordim_flag == 'avamerg':
        if inter_print: print (f'{dataset}: best emotion_acc: %.4f; best hitrate: %.4f; best dist1: %.4f; best dist2: %.4f (epoch: {best_epoch})' %(best_score1, best_score2, best_score3, best_score4))
    elif disordim_flag == 'mintrec':
        if inter_print: print (f'{dataset}: best intent_acc: %.2f; best weighted_f1: %.2f; best weighted_precision: %.2f (epoch: {best_epoch})' %(best_score1, best_score2, best_score3))
    elif disordim_flag == 'binary_text':
        if inter_print: print (f'{dataset}: best acc: {best_score1}; best WF1: {best_score2}; best WP: {best_score3}; best precision: {best_score4}; best F1: {best_score5}; best recall: {best_score6} (epoch: {best_epoch})')
    # return the best scores and best epoch
    return best_score1, best_score2, best_score3, best_score4, best_epoch



def func_return_scores_one(modelname=None, dataset_candidates='merunibench'):
    ## => (process datasets)
    if dataset_candidates=='merunibench':
        process_datasets = ["mer2023", "mer2024", "meld", "iemocapfour", "cmumosi", "cmumosei", "sims", "simsv2","ovmerd"]
    elif dataset_candidates=='mer2025ov':
        process_datasets = ['mer2025ov']
    elif dataset_candidates=='avamerg':
        process_datasets = ['avamerg']
    elif dataset_candidates=='intent':
        process_datasets = ['mintrec','mintrec2']
    elif dataset_candidates=='mustard_urfunny':
        process_datasets = ['mustard', 'urfunny']
    else:
        names = dataset_candidates.split(',')
        process_datasets = names

    print_per_dataset, avg_score = [], []
    best_epochs = [] 
    for dataset in process_datasets:
        process_root = f"output/results-{dataset}/{modelname}"
        score1, score2, score3, score4, best_epoch = main_zeroshot_scores(process_root, debug=False, test_epochs='', inter_print=True)
        print_per_dataset.extend([score1])
        if dataset_candidates == 'avamerg':
            print_per_dataset.extend([score1,score2, score3, score4])
        best_epochs.append(best_epoch)
        avg_score.append(score1)
    # append a avg value for ranking
    avg_score = np.mean(avg_score)
    print_per_dataset.append(avg_score)
    best_epochs.append(-1)
    formatted_results = []
    for i, (score, epoch) in enumerate(zip(print_per_dataset, best_epochs)):
        if epoch == -1: 
            formatted_results.append("| %.2f" % (score * 100))
        else:
            formatted_results.append("| %.2f(epoch %d)" % (score * 100, epoch))
    return formatted_results, avg_score


if __name__ == "__main__":
    """

    benchmark candidate :
                "merunibench"    -> inference_data (MER2023/MER2024/MELD/IEMOCAPFour/
                                   CMUMOSI/CMUMOSEI/SIMS/SIMSv2/OVMERD) (MSA, B-MER, OV-MER Task)
                "intent"         -> MIntRec / MIntRec2 (MIR Task)
                "mustard_urfunny"-> Mustard / URFunny (MSU,MHU Task)
                single benchmark usecase: "mustard", "urfunny", "mintrec", ...
    """
    parser = argparse.ArgumentParser(description='OneEmo evaluation (score-only)')
    parser.add_argument('--modelname', type=str, default='rl_v1',
                        help='model name (the result of output/results-<dataset>/<modelname> )')
    parser.add_argument('--datasets', type=str, default='mustard_urfunny',
                        help="benchmark collection: merunibench | intent | mustard_urfunny | "
                             "or single benchmark name")
    args = parser.parse_args()

    print_per_dataset, avg_score = func_return_scores_one(
        modelname=args.modelname, dataset_candidates=args.datasets)
    print(args.modelname, " ".join(print_per_dataset))
