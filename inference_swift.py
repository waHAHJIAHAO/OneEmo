import os
# os.environ['SWIFT_DEBUG'] = '1'
os.environ['IMAGE_MAX_TOKEN_NUM'] = '1024'
os.environ['VIDEO_MAX_TOKEN_NUM'] = '128'
os.environ['FPS_MAX_FRAMES'] = '16'

import argparse
import importlib.util
import json
import numpy as np
import torch
from tqdm import tqdm
from swift import get_model_processor, get_template
from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
from config import (
    MODEL_PATH as DEFAULT_MODEL_PATH,
    RESULTS_DIR,
    CHECKPOINT_NAME,
    DATASET_COLLECTIONS
)
from dataset import VideoDataset

# support task: msa,mer,ovmer,mir,msd,mhd,erg
# for zeroshot, we use ovmer
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

OUTPUT_BASE_DIR = '/path/to/your/OneEmo/output'
# OUTPUT_BASE_DIR = '/path/to/your/OneEmo/results'

SUPPORTED_MODEL_TYPE_ALIASES = {
    'qwen2_5_vl': 'qwen2_5_vl',
    'qwen3_vl': 'qwen3_vl',
    'qwen3_5': 'qwen3_5',
}

SUPPORTED_ARCHITECTURES = {
    'Qwen2_5_VLForConditionalGeneration': 'qwen2_5_vl',
    'Qwen3VLForConditionalGeneration': 'qwen3_vl',
    'Qwen3_5ForConditionalGeneration': 'qwen3_5',
}


def patch_qwen25_vl_image_processor_compat() -> None:
    try:
        from transformers.models.auto import image_processing_auto
    except Exception:
        return

    get_processor_class = image_processing_auto.get_image_processor_class_from_name
    if get_processor_class('Qwen2_5_VLImageProcessor') is not None:
        return

    def get_processor_class_with_qwen25_compat(class_name: str):
        if class_name == 'Qwen2_5_VLImageProcessor':
            class_name = 'Qwen2VLImageProcessor'
        elif class_name == 'Qwen2_5_VLImageProcessorFast':
            class_name = 'Qwen2VLImageProcessorFast'
        return get_processor_class(class_name)

    image_processing_auto.get_image_processor_class_from_name = get_processor_class_with_qwen25_compat


def resolve_model_type(model_path: str) -> str:
    config_path = os.path.join(model_path, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Model config not found: {config_path}')

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    raw_model_type = config.get('model_type')
    if raw_model_type in SUPPORTED_MODEL_TYPE_ALIASES:
        return SUPPORTED_MODEL_TYPE_ALIASES[raw_model_type]

    architectures = config.get('architectures') or []
    if architectures:
        architecture = architectures[0]
        if architecture in SUPPORTED_ARCHITECTURES:
            return SUPPORTED_ARCHITECTURES[architecture]

    raise ValueError(
        f'Unsupported model config for {model_path}. '
        f'model_type={raw_model_type!r}, architectures={architectures!r}. '
        f'Expected model_type in {sorted(SUPPORTED_MODEL_TYPE_ALIASES)} or '
        f'architectures containing one of {sorted(SUPPORTED_ARCHITECTURES)}.'
    )


def get_available_attn_impl(preferred_attn_impl: str) -> str:
    if preferred_attn_impl != 'flash_attn':
        return preferred_attn_impl

    if importlib.util.find_spec('flash_attn') is not None:
        return preferred_attn_impl

    print('flash_attn is not installed; falling back to sdpa attention.')
    return 'sdpa'


def ensure_runtime_support(resolved_model_type: str) -> None:
    if resolved_model_type != 'qwen3_vl':
        return

    try:
        from transformers import Qwen3VLForConditionalGeneration  # noqa: F401
        from swift.model import MODEL_MAPPING  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            'The local code now recognizes Qwen3-VL, but the current runtime does not appear '
            f'to support it yet ({e}). Please upgrade to a Qwen3-VL-capable stack, '
            'for example `transformers>=4.57` and a recent `ms-swift` release that includes '
            '`model_type=qwen3_vl`.'
        )


def inference_video(
    video_path: str,
    transcript: str,
    dataset_name: str,
    engine: TransformersEngine,
    request_config: RequestConfig
) -> str:
    task = dataset_to_task.get(dataset_name, 'mer')

    if task == 'esc':
        infer_request = InferRequest(
            messages=[{
                "role": "user",
                "content": transcript
            }]
        )
        resp_list = engine.infer([infer_request], request_config=request_config)
        return resp_list[0].choices[0].message.content

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

    infer_request = InferRequest(
        messages=[{
            "role": "user",
            "content": f'{prompt}'
        }],
        videos=[video_path]
    )

    resp_list = engine.infer([infer_request], request_config=request_config)
    response = resp_list[0].choices[0].message.content
    return response


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
    parser = argparse.ArgumentParser(description="Inference script for video emotion recognition (Swift)")
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
        '--think',
        type=lambda x: x.lower() == 'true',
        default=False,
        help='Enable thinking mode (true/false)'
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
    parser.add_argument(
        '--attn_impl',
        type=str,
        default='flash_attn',
        choices=['flash_attn', 'sdpa', 'eager'],
        help='Attention implementation for model loading'
    )
    args = parser.parse_args()

    dataset_names = []
    if args.datasets:
        for arg in args.datasets:
            if arg in DATASET_COLLECTIONS:
                dataset_names.extend(DATASET_COLLECTIONS[arg])
            else:
                dataset_names.append(arg)
    else:
        dataset_names = None

    model_path = args.model_path.rstrip('/')
    model_name = os.path.basename(model_path)
    verbose = args.verbose
    force = args.force
    think = args.think
    attn_impl = get_available_attn_impl(args.attn_impl)
    resolved_model_type = resolve_model_type(model_path)
    ensure_runtime_support(resolved_model_type)

    print(f'[INFO] resolved_model_type: {resolved_model_type}')

    if resolved_model_type == 'qwen2_5_vl':
        patch_qwen25_vl_image_processor_compat()

    model, processor = get_model_processor(
        model_path,
        model_type=resolved_model_type,
        torch_dtype=torch.bfloat16,
        attn_impl=attn_impl,
    )
    template = get_template(processor, enable_thinking=think)
    engine = TransformersEngine(model, template=template)
    # repetition_penalty=1.5
    request_config = RequestConfig(max_tokens=2048, temperature=0.7)

    dataset = VideoDataset(dataset_names)
    all_datasets = dataset.get_all_datasets()

    for dataset_name, samples in all_datasets.items():
        if not samples:
            print(f"No samples found for {dataset_name}, skipping...")
            continue

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
                result = inference_video(
                    video_path,
                    transcript,
                    dataset_name,
                    engine,
                    request_config
                )
                name2reason[sample_name] = result

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
