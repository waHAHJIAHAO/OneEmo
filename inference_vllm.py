import os
import numpy as np
import argparse
from tqdm import tqdm
from openai import OpenAI
from config import (
    MODEL_PATH as DEFAULT_MODEL_PATH, 
    RESULTS_DIR, 
    CHECKPOINT_NAME,
    DATASET_COLLECTIONS
)
from dataset import VideoDataset

os.environ['IMAGE_MAX_TOKEN_NUM'] = '1024'
os.environ['VIDEO_MAX_TOKEN_NUM'] = '128'
os.environ['FPS_MAX_FRAMES'] = '16'
OUTPUT_BASE_DIR = '/path/to/your/OneEmo/output'

dataset_to_task = {
    'OVMERD': 'ovmer',
    'MER2023': 'ovmer',
    'MER2024': 'ovmer',
    'MELD': 'ovmer',
    'IEMOCAPFour': 'ovmer',
    'CMUMOSI': 'ovmer',
    'CMUMOSEI': 'ovmer',
    'SIMS': 'ovmer',
    'SIMSv2': 'ovmer',
    'MIntRec': 'mir',
    'MIntRec2': 'mir',
    'URFunny': 'mhd',
    'Mustard': 'msd',
    'AvaMERG': 'erg',
    'Openr1psy': 'esc',
}

MIR_CANDIDATE_LABELS = {
    'MIntRec': 'complain, praise, apologise, thank, criticize, agree, taunt, flaunt, joke, oppose, comfort, care, inform, advise, arrange, introduce, leave, prevent, greet, ask for help',
    'MIntRec2': 'acknowledge, advise, agree, apologise, arrange, ask for help, asking for opinions, care, comfort, complain, confirm, criticize, doubt, emphasize, explain, flaunt, greet, inform, introduce, invite, joke, leave, oppose, plan, praise, prevent, refuse, taunt, thank, warn',
}

try:
    from prompts.mer import get_mer_prompt
    from prompts.mir import get_mir_prompt
    from prompts.ovmer import get_ovmer_prompt
    from prompts.msa import get_msa_prompt
    from prompts.mhd import get_mhd_prompt
    from prompts.msd import get_msd_prompt
    from prompts.erg import get_erg_prompt
    from prompts.esc import get_esc_prompt
    prompt_functions = {
        'mer': get_mer_prompt,
        'mir': get_mir_prompt,
        'ovmer': get_ovmer_prompt,
        'msa': get_msa_prompt,
        'mhd': get_mhd_prompt,
        'msd': get_msd_prompt,
        'erg': get_erg_prompt,
        'esc': get_esc_prompt,
    }
except ImportError as e:
    print(f"Warning: Failed to import prompt modules: {e}")
    prompt_functions = {}

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1"
)


def inference_video(video_path: str, transcript: str, model_path: str, dataset_name: str) -> str:
    # 根据数据集获取任务类型
    task = dataset_to_task.get(dataset_name, 'mer')

    if task == 'esc':
        messages = [
            {
                "role": "user",
                "content": transcript
            }
        ]

        chat_response = client.chat.completions.create(
            model=model_path,
            messages=messages,
            max_tokens=2048,
            temperature=0.7,
            top_p=0.9, # presence_penalty=1.5,
            extra_body={
                "top_k": 50,
                "chat_template_kwargs": {"enable_thinking": True}
            },
        )

        message = chat_response.choices[0].message
        reasoning = getattr(message, 'reasoning', '')
        content = message.content

        if reasoning:
            return f"<think>{reasoning}</think>{content}"
        return content
    
    # 获取对应的提示词
    if task == 'mir':
        candidate_labels = MIR_CANDIDATE_LABELS.get(dataset_name)
        if candidate_labels is None:
            raise ValueError(f'Missing candidate labels for MIR dataset: {dataset_name}')
        base_prompt = prompt_functions[task](candidate_labels)
        if transcript:
            prompt = f"{base_prompt}\nHere's the character says in the video: {transcript}"
        else:
            prompt = base_prompt
    elif task in prompt_functions:
        base_prompt = prompt_functions[task]()
        if task == 'erg' and transcript:
            prompt = f"{base_prompt}\n{transcript}"
        elif transcript:
            prompt = f"{base_prompt}\nHere is what the character says: {transcript}"
        else:
            prompt = base_prompt
    else:
        base_prompt = "Recognize the emotions of the characters in the video and explain the reasons."
        if transcript:
            prompt = f"{base_prompt}\nHere is what the character says: {transcript}"
        else:
            prompt = base_prompt

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {
                        "url": f"file://{video_path}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]

    chat_response = client.chat.completions.create(
        model=model_path,
        messages=messages,
        max_tokens=2048,
        temperature=0.7,
        top_p=0.9, #presence_penalty=1.5,
        extra_body={
            "top_k": 50,
            "chat_template_kwargs": {"enable_thinking": True}
        },
    )

    # 获取 reasoning 和 content
    message = chat_response.choices[0].message
    reasoning = getattr(message, 'reasoning', '')
    content = message.content
    
    # 组合推理过程和最终结果
    if reasoning:
        return f"<think>{reasoning}</think>{content}"
    else:
        return content


def get_save_path(dataset_name: str, model_name: str) -> str:
    save_dir = os.path.join(OUTPUT_BASE_DIR, f'results-{dataset_name.lower()}', model_name)
    return os.path.join(save_dir, CHECKPOINT_NAME)


def save_results(dataset_name: str, name2reason: dict, model_name: str):
    save_dir = os.path.join(OUTPUT_BASE_DIR, f'results-{dataset_name.lower()}', model_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, CHECKPOINT_NAME)
    np.savez_compressed(save_path, name2reason=name2reason)
    print(f"Saved results to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Inference script for video emotion recognition")
    parser.add_argument(
        '--datasets', 
        nargs='+', 
        help='List of datasets or dataset collections to process'
    )
    parser.add_argument(
        '--model_path', 
        type=str, 
        default=DEFAULT_MODEL_PATH, 
        help='Path to the model checkpoint'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true', 
        help='Print results for every sample'
    )
    parser.add_argument(
        '--force', 
        action='store_true', 
        help='Force re-inference even if results already exist'
    )
    args = parser.parse_args()

    # 确定要处理的数据集
    dataset_names = []
    if args.datasets:
        for arg in args.datasets:
            if arg in DATASET_COLLECTIONS:
                # 如果是数据集集合，添加集合中的所有数据集
                dataset_names.extend(DATASET_COLLECTIONS[arg])
            else:
                # 否则直接添加数据集名称
                dataset_names.append(arg)
    else:
        # 默认处理所有数据集
        dataset_names = None

    # 获取模型路径和模型名称
    model_path = args.model_path.rstrip('/')
    model_name = os.path.basename(model_path)
    verbose = args.verbose
    force = args.force

    dataset = VideoDataset(dataset_names)
    all_datasets = dataset.get_all_datasets()

    for dataset_name, samples in all_datasets.items():
        if not samples:
            print(f"No samples found for {dataset_name}, skipping...")
            continue
        
        # 检查是否已经有推理结果
        save_path = get_save_path(dataset_name, model_name)
        if os.path.exists(save_path) and not force:
            print(f"\nResults already exist for {dataset_name}: {save_path}")
            print("Skipping inference. Use --force to re-run.")
            continue

        print(f"\nProcessing dataset: {dataset_name} ({len(samples)} samples)")
        name2reason = {}
        printed_sample = False

        for sample_name, sample_data in tqdm(samples.items(), desc=dataset_name):
            try:
                video_path = sample_data['video_path']
                transcript = sample_data['transcription']
                result = inference_video(video_path, transcript, model_path, dataset_name)
                name2reason[sample_name] = result
                
                # 根据verbose参数决定是否打印结果
                if verbose or not printed_sample:
                    print(f"\nSample: {sample_name}")
                    print(f"\n###Assistant: {result}")
                    printed_sample = True
            except Exception as e:
                error_msg = f"Error processing {sample_name}: {e}"
                print(error_msg)
                name2reason[sample_name] = f"ERROR: {str(e)}"

        save_results(dataset_name, name2reason, model_name)


if __name__ == "__main__":
    main()
