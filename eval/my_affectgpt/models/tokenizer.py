"""
Adapted from salesforce@LAVIS. Below is the original copyright:
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import os
from transformers import AutoTokenizer
import config


## 在 AffectGPT 中，每个 LLM 都需要自己的 'eos', 'pad', 'bos'；否则模型会报错
def load_tokenizer_from_LLM(model_name):
    if model_name in ['Baichuan2']:
        tokenizer = AutoTokenizer.from_pretrained(config.PATH_TO_LLM[model_name], use_fast=False, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.PATH_TO_LLM[model_name], use_fast=False)
    if model_name in ['Qwen2', 'Qwen25']: tokenizer.bos_token='<|im_start|>'
    tokenizer.pad_token = tokenizer.eos_token # 看看如果全设置成这样子会有什么影响？ vicuna, llama2, llama3
    tokenizer.add_tokens([config.DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    tokenizer.add_tokens([config.DEFAULT_AUDIO_PATCH_TOKEN], special_tokens=True)
    tokenizer.add_tokens([config.DEFAULT_FRAME_PATCH_TOKEN], special_tokens=True)
    tokenizer.add_tokens([config.DEFAULT_FACE_PATCH_TOKEN],  special_tokens=True)
    tokenizer.add_tokens([config.DEFAULT_MULTI_PATCH_TOKEN], special_tokens=True)
    return tokenizer
