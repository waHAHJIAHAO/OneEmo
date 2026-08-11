import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError
from statsmodels.stats.inter_rater import fleiss_kappa

try:
    import krippendorff
except ImportError:  # pragma: no cover - runtime dependency check
    krippendorff = None

from prompt import ESC_EVAL_PROMPT, ERG_EVAL_PROMPT


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / "eval" / ".env"
ERG_DATA_PATH = PROJECT_ROOT / "datas" / "eval" / "testset_avamerg.json"
ESC_DATA_PATH = PROJECT_ROOT / "datas" / "eval" / "test_openr1psy.json"
RESULTS_ROOT = Path("/public/home/lianzheng/hjh/AffectGPT/AffectGPT/output")
OUTPUT_ROOT = PROJECT_ROOT / "eval" / "output" / "quality_eval"

RETRYABLE_ERRORS = (APIConnectionError, APIError, APITimeoutError, RateLimitError)


@dataclass
class TaskConfig:
    task_name: str
    dataset_path: Path
    results_dir: Path
    dimensions: Sequence[str]
    prompt_template: str


@dataclass
class JudgeSpec:
    judge_id: str
    provider: str
    model_name: str
    client: OpenAI


@dataclass
class SampleRecord:
    sample_index: int
    sample_key: str
    context: str
    chat_history: str
    current_input: str
    model_response: str


TASKS: Dict[str, TaskConfig] = {
    "erg": TaskConfig(
        task_name="erg",
        dataset_path=ERG_DATA_PATH,
        results_dir=RESULTS_ROOT / "results-avamerg",
        dimensions=("Empathy", "Coherence", "Informativeness"),
        prompt_template=ERG_EVAL_PROMPT,
    ),
    "esc": TaskConfig(
        task_name="esc",
        dataset_path=ESC_DATA_PATH,
        results_dir=RESULTS_ROOT / "results-openr1psy",
        dimensions=("Empathy", "Skill", "Overall"),
        prompt_template=ESC_EVAL_PROMPT,
    ),
}


def load_env(env_path: Path) -> Dict[str, str]:
    env_dict: Dict[str, str] = {}
    with env_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env_dict[key.strip()] = value.strip().strip('"').strip("'")
    return env_dict


def get_env_value(env_dict: Dict[str, str], *candidates: str) -> Optional[str]:
    for candidate in candidates:
        value = env_dict.get(candidate)
        if value:
            return value
    lowered = {key.lower(): value for key, value in env_dict.items()}
    for candidate in candidates:
        value = lowered.get(candidate.lower())
        if value:
            return value
    return None


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip().strip('"').strip("'")


def extract_memory(text: str) -> str:
    match = re.search(r"<memory>(.*?)</memory>", text, re.S)
    return normalize_text(match.group(1)) if match else ""


def extract_erg_context(text: str) -> str:
    patterns = [
        r"The context from the speaker is \((.*?)\)\.\s+The topic of the conversation is",
        r"The context from the speaker is \((.*?)\)\.",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.S)
        if match:
            return normalize_text(match.group(1))
    raise ValueError("Failed to parse ERG context from JSON content.")


def extract_erg_current_input(text: str) -> str:
    match = re.search(r"now speaker say:\s*(.*)$", text, re.S)
    if not match:
        raise ValueError("Failed to parse ERG current input from JSON content.")
    return normalize_text(match.group(1))


def extract_esc_current_input(text: str) -> str:
    patterns = [
        r'now,\s*the patient say that:\s*"(.*)"\s*$',
        r"now,\s*the patient say that:\s*(.*)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.S)
        if match:
            return normalize_text(match.group(1))
    raise ValueError("Failed to parse ESC current input from JSON content.")


def load_task_json_samples(task_name: str) -> List[Dict[str, str]]:
    task_config = TASKS[task_name]
    with task_config.dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list in {task_config.dataset_path}, got {type(data).__name__}.")

    parsed_samples: List[Dict[str, str]] = []
    for item in data:
        messages = item.get("messages") or []
        if not messages:
            raise ValueError(f"Missing messages in {task_config.dataset_path}.")
        user_content = messages[0].get("content", "")
        chat_history = extract_memory(user_content)
        if task_name == "erg":
            parsed_samples.append(
                {
                    "context": extract_erg_context(user_content),
                    "chat_history": chat_history,
                    "current_input": extract_erg_current_input(user_content),
                }
            )
        else:
            parsed_samples.append(
                {
                    "context": "",
                    "chat_history": chat_history,
                    "current_input": extract_esc_current_input(user_content),
                }
            )
    return parsed_samples


def resolve_latest_checkpoint(task_config: TaskConfig, model_name: str) -> Path:
    model_dir = task_config.results_dir / model_name
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    candidates = [path for path in model_dir.glob("*.npz") if not path.name.endswith("-openset.npz")]
    if not candidates:
        raise FileNotFoundError(f"No valid checkpoint npz found under {model_dir}")

    def checkpoint_sort_key(path: Path) -> Tuple[int, str]:
        match = re.search(r"checkpoint_(\d+)", path.name)
        step = int(match.group(1)) if match else -1
        return step, path.name

    return max(candidates, key=checkpoint_sort_key)


def load_npz_dicts(npz_path: Path) -> Dict[str, Dict[str, Any]]:
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as exc:  # pragma: no cover - runtime file compatibility
        raise RuntimeError(f"Failed to load npz {npz_path}: {exc}") from exc

    result: Dict[str, Dict[str, Any]] = {}
    for key in data.files:
        value = data[key]
        if getattr(value, "shape", None) == ():
            result[key] = value.item()
    return result


def extract_model_response(raw_text: Any) -> str:
    text = "" if raw_text is None else str(raw_text)
    answer_match = re.search(r"<answer>([\s\S]*?)</answer>", text)
    if answer_match:
        return normalize_text(answer_match.group(1))
    without_think = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.I).strip()
    without_think = re.sub(r"<thinking>[\s\S]*?</thinking>", "", without_think, flags=re.I).strip()
    return normalize_text(without_think or text)


def align_samples(
    task_name: str,
    parsed_json_samples: Sequence[Dict[str, str]],
    npz_payload: Dict[str, Dict[str, Any]],
) -> List[SampleRecord]:
    if "name2reason" not in npz_payload:
        raise KeyError("NPZ missing required key: name2reason")

    name2reason = npz_payload["name2reason"]
    if not isinstance(name2reason, dict):
        raise TypeError("name2reason is not a dictionary")

    if len(parsed_json_samples) != len(name2reason):
        raise ValueError(
            f"Sample count mismatch for task {task_name}: "
            f"json={len(parsed_json_samples)} npz={len(name2reason)}"
        )

    name2current_input = npz_payload.get("name2current_input")
    aligned: List[SampleRecord] = []
    for sample_index, ((sample_key, raw_output), parsed_sample) in enumerate(
        zip(name2reason.items(), parsed_json_samples)
    ):
        model_response = extract_model_response(raw_output)
        current_input = parsed_sample["current_input"]
        if isinstance(name2current_input, dict) and sample_key in name2current_input:
            npz_current = normalize_text(str(name2current_input[sample_key]))
            if npz_current and npz_current != normalize_text(current_input):
                raise ValueError(
                    f"Current input mismatch at sample {sample_index} ({sample_key}): "
                    f"json={current_input!r} npz={npz_current!r}"
                )

        aligned.append(
            SampleRecord(
                sample_index=sample_index,
                sample_key=str(sample_key),
                context=parsed_sample["context"],
                chat_history=parsed_sample["chat_history"],
                current_input=current_input,
                model_response=model_response,
            )
        )
    return aligned


def build_judge_specs(env_path: Path, provider: str, judge_llm: str) -> List[JudgeSpec]:
    env_dict = load_env(env_path)
    judge_catalog = {
        "openai": {
            "provider": "openai",
            "model_name": get_env_value(env_dict, "OpenAI_MODEL_NAME"),
            "base_url": get_env_value(env_dict, "OpenAI_BASE_URL"),
            "api_key": get_env_value(env_dict, "OpenAI_API_KEY"),
        },
        "mimo": {
            "provider": "mimo",
            "model_name": get_env_value(env_dict, "MIMO_EVAL_MODEL_NAME"),
            "base_url": get_env_value(env_dict, "MIMO_BASE_URL"),
            "api_key": get_env_value(env_dict, "MIMO_API_KEY"),
        },
        "deepseek": {
            "provider": "deepseek",
            "model_name": get_env_value(env_dict, "DeepSeek_MODEL_NAME"),
            "base_url": get_env_value(env_dict, "DeepSeek_BASE_URL"),
            "api_key": get_env_value(env_dict, "DeepSeek_API_KEY"),
        },
    }

    if provider == "all":
        allowed_providers = {"openai", "mimo", "deepseek"}
    else:
        allowed_providers = {provider}

    if judge_llm == "all":
        selected_ids = [
            judge_id
            for judge_id, config in judge_catalog.items()
            if config["provider"] in allowed_providers
        ]
    else:
        if judge_llm not in judge_catalog:
            raise ValueError(f"Unsupported judge_llm: {judge_llm}")
        if judge_catalog[judge_llm]["provider"] not in allowed_providers:
            raise ValueError(f"judge_llm={judge_llm} is not compatible with provider={provider}")
        selected_ids = [judge_llm]

    specs: List[JudgeSpec] = []
    for judge_id in selected_ids:
        config = judge_catalog[judge_id]
        if not config["model_name"] or not config["base_url"] or not config["api_key"]:
            raise ValueError(f"Missing API configuration for judge: {judge_id}")
        base_url = str(config["base_url"]).rstrip("/")
        if config["provider"] in {"openai", "mimo"} and not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        client = OpenAI(api_key=str(config["api_key"]), base_url=base_url, timeout=120.0)
        specs.append(
            JudgeSpec(
                judge_id=judge_id,
                provider=str(config["provider"]),
                model_name=str(config["model_name"]),
                client=client,
            )
        )
    return specs


def build_task_prompt(task_config: TaskConfig, sample: SampleRecord) -> str:
    prompt = task_config.prompt_template
    replacements = {
        "{{context}}": sample.context or "",
        "{{chat_history}}": sample.chat_history or "",
        "{{seeker_utterance}}": sample.current_input or "",
        "{{patient_utterance}}": sample.current_input or "",
        "{{model_response}}": sample.model_response or "",
    }
    for src, dst in replacements.items():
        prompt = prompt.replace(src, dst)
    strict_suffix = [
        "",
        "Return exactly one score for every required dimension.",
        "Do not omit any dimension.",
        "Do not explain your reasoning.",
    ]
    for dimension in task_config.dimensions:
        strict_suffix.append(f"{dimension}: <integer 1-5>")
    strict_suffix.append("Output exactly these lines and nothing else.")
    return prompt + "\n" + "\n".join(strict_suffix)


def build_retry_prompt(base_prompt: str, invalid_output: str, dimensions: Sequence[str]) -> str:
    lines = [
        base_prompt,
        "",
        "Your previous output was invalid because it did not include all required dimensions.",
        "Previous invalid output:",
        invalid_output or "<empty>",
        "",
        "Correct it now.",
        "Return exactly these lines and nothing else:",
    ]
    for dimension in dimensions:
        lines.append(f"{dimension}: <integer 1-5>")
    return "\n".join(lines)


def get_output_dir(task_name: str, model_name: str, checkpoint_path: Path) -> Path:
    return OUTPUT_ROOT / task_name / model_name / checkpoint_path.stem


def get_progress_paths(task_name: str, model_name: str, checkpoint_path: Path) -> Tuple[Path, Path, Path]:
    output_dir = get_output_dir(task_name, model_name, checkpoint_path)
    return (
        output_dir / "sample_scores.jsonl",
        output_dir / "summary.json",
        output_dir / "progress.json",
    )


def extract_judge_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError(f"Judge response missing choices: {response}")
    choice = choices[0]
    message = choice.message
    content = getattr(message, "content", None) or ""
    reasoning_content = getattr(message, "reasoning_content", None) or ""
    refusal = getattr(message, "refusal", None) or ""

    if content and str(content).strip():
        return str(content)
    if reasoning_content and str(reasoning_content).strip():
        return str(reasoning_content)
    if refusal and str(refusal).strip():
        return str(refusal)
    return ""


def summarize_empty_judge_response(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        return f"choices={choices!r}, response={response!r}"
    choice = choices[0]
    message = choice.message
    finish_reason = getattr(choice, "finish_reason", None)
    content = getattr(message, "content", None)
    reasoning_content = getattr(message, "reasoning_content", None)
    refusal = getattr(message, "refusal", None)
    return (
        f"finish_reason={finish_reason!r}, "
        f"content={content!r}, "
        f"reasoning_content={reasoning_content!r}, "
        f"refusal={refusal!r}"
    )


def call_judge_model(prompt: str, judge: JudgeSpec, dimensions: Sequence[str]) -> Tuple[str, Dict[str, int]]:
    last_error: Optional[Exception] = None
    invalid_output = ""
    for attempt in range(1, 4):
        try:
            request_prompt = (
                build_retry_prompt(prompt, invalid_output, dimensions)
                if invalid_output
                else prompt
            )
            request_kwargs: Dict[str, Any] = {
                "model": judge.model_name,
                "messages": [{"role": "user", "content": request_prompt}],
                "stream": False,
            }
            if judge.provider == "mimo":
                request_kwargs["max_completion_tokens"] = 1024
                request_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            else:
                request_kwargs["max_tokens"] = 512

            if judge.provider == "deepseek":
                request_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            elif judge.provider != "mimo":
                request_kwargs["temperature"] = 0

            response = judge.client.chat.completions.create(
                **request_kwargs,
            )
            content = extract_judge_text(response)
            if not content.strip():
                raise ValueError(
                    "Judge returned empty text: " + summarize_empty_judge_response(response)
                )
            parsed = parse_dimension_scores(content, dimensions)
            return content, parsed
        except RETRYABLE_ERRORS as exc:
            last_error = exc
        except ValueError as exc:
            last_error = exc
            invalid_output = str(exc).split("judge output:", 1)[-1].strip() if "judge output:" in str(exc) else ""
        if attempt < 3:
            time.sleep(1.5 * attempt)
    raise RuntimeError(
        f"Judge {judge.judge_id} ({judge.model_name}) failed after 3 attempts: {last_error}"
    ) from last_error


def parse_dimension_scores(text: str, dimensions: Sequence[str]) -> Dict[str, int]:
    parsed: Dict[str, int] = {}
    for dimension in dimensions:
        pattern = rf"{re.escape(dimension)}\s*:\s*(?:score\s*\(?\s*)?([1-5])"
        match = re.search(pattern, text, re.I)
        if not match:
            raise ValueError(f"Failed to parse dimension '{dimension}' from judge output: {text}")
        parsed[dimension] = int(match.group(1))
    return parsed


def median_of_three(values: Sequence[int]) -> int:
    if len(values) != 3:
        raise ValueError(f"Expected exactly 3 values, got {len(values)}")
    ordered = sorted(values)
    return ordered[1]


def compute_aggregated_scores(
    sample_outputs: List[Dict[str, Any]],
    dimensions: Sequence[str],
) -> Tuple[List[Dict[str, int]], Dict[str, float]]:
    aggregated_samples: List[Dict[str, int]] = []
    for sample_output in sample_outputs:
        per_judge = sample_output["parsed_scores"]
        aggregated = {
            dimension: median_of_three([scores[dimension] for scores in per_judge.values()])
            for dimension in dimensions
        }
        aggregated_samples.append(aggregated)
        sample_output["aggregated_scores"] = aggregated

    dimension_means = {
        dimension: float(np.mean([item[dimension] for item in aggregated_samples]))
        for dimension in dimensions
    }
    return aggregated_samples, dimension_means


def compute_alpha(
    sample_outputs: List[Dict[str, Any]],
    judge_ids: Sequence[str],
    dimensions: Sequence[str],
) -> Tuple[Dict[str, float], float]:
    if krippendorff is None:
        raise ImportError(
            "The 'krippendorff' package is required but not installed. "
            "Please install it before running this script."
        )

    alpha_by_dimension: Dict[str, float] = {}
    for dimension in dimensions:
        reliability_data = np.array(
            [
                [sample_output["parsed_scores"][judge_id][dimension] for sample_output in sample_outputs]
                for judge_id in judge_ids
            ]
        )
        alpha_value = krippendorff.alpha(
            reliability_data=reliability_data,
            level_of_measurement="ordinal",
        )
        alpha_by_dimension[dimension] = float(alpha_value)
    alpha_mean = float(np.mean(list(alpha_by_dimension.values())))
    return alpha_by_dimension, alpha_mean


def compute_agreement_metrics(
    sample_outputs: List[Dict[str, Any]],
    judge_ids: Sequence[str],
    dimensions: Sequence[str],
) -> Tuple[
    Dict[str, float],
    Dict[str, float],
    float,
    float,
    Dict[str, float],
    float,
]:
    full_agreement_by_dimension: Dict[str, float] = {}
    pairwise_agreement_by_dimension: Dict[str, float] = {}
    randolph_kappa_by_dimension: Dict[str, float] = {}

    for dimension in dimensions:
        full_agreement_count = 0
        pairwise_agreement_total = 0.0
        reliability_rows: List[List[int]] = []

        for sample_output in sample_outputs:
            values = [sample_output["parsed_scores"][judge_id][dimension] for judge_id in judge_ids]
            if values[0] == values[1] == values[2]:
                full_agreement_count += 1
            agree_pairs = sum(
                values[a] == values[b]
                for a, b in ((0, 1), (0, 2), (1, 2))
            )
            pairwise_agreement_total += agree_pairs / 3.0

            counts = [0, 0, 0, 0, 0]
            for value in values:
                counts[int(value) - 1] += 1
            reliability_rows.append(counts)

        full_agreement_by_dimension[dimension] = full_agreement_count / len(sample_outputs)
        pairwise_agreement_by_dimension[dimension] = pairwise_agreement_total / len(sample_outputs)
        randolph_kappa_by_dimension[dimension] = float(
            fleiss_kappa(np.asarray(reliability_rows), method="randolph")
        )

    full_agreement_mean = float(np.mean(list(full_agreement_by_dimension.values())))
    pairwise_agreement_mean = float(np.mean(list(pairwise_agreement_by_dimension.values())))
    return (
        full_agreement_by_dimension,
        pairwise_agreement_by_dimension,
        full_agreement_mean,
        pairwise_agreement_mean,
        randolph_kappa_by_dimension,
        float(np.mean(list(randolph_kappa_by_dimension.values()))),
    )


def make_sample_output(sample: SampleRecord) -> Dict[str, Any]:
    return {
        "sample_index": sample.sample_index,
        "sample_key": sample.sample_key,
        "context": sample.context,
        "chat_history": sample.chat_history,
        "current_input": sample.current_input,
        "model_response": sample.model_response,
        "judge_models": {},
        "raw_judge_outputs": {},
        "parsed_scores": {},
        "aggregated_scores": None,
    }


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_progress(progress_path: Path) -> Dict[str, Any]:
    if not progress_path.exists():
        return {}
    with progress_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_progress(
    progress_path: Path,
    task_name: str,
    model_name: str,
    checkpoint_path: Path,
    judges: Sequence[JudgeSpec],
    sample_outputs: Sequence[Dict[str, Any]],
) -> None:
    progress = {
        "task": task_name,
        "model": model_name,
        "checkpoint_path": str(checkpoint_path),
        "judge_models": {judge.judge_id: judge.model_name for judge in judges},
        "completed_samples": len(sample_outputs),
        "last_sample_index": sample_outputs[-1]["sample_index"] if sample_outputs else None,
        "updated_at": datetime.now().isoformat(),
    }
    with progress_path.open("w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def is_sample_complete(sample_output: Dict[str, Any], judge_ids: Sequence[str]) -> bool:
    parsed_scores = sample_output.get("parsed_scores", {})
    return all(judge_id in parsed_scores for judge_id in judge_ids)


def persist_progress_snapshot(
    sample_scores_path: Path,
    progress_path: Path,
    task_name: str,
    model_name: str,
    checkpoint_path: Path,
    judges: Sequence[JudgeSpec],
    sample_outputs: Sequence[Dict[str, Any]],
) -> None:
    write_jsonl(sample_scores_path, sample_outputs)
    save_progress(
        progress_path,
        task_name,
        model_name,
        checkpoint_path,
        judges,
        sample_outputs,
    )


def print_sample_header(sample: SampleRecord) -> None:
    print(f"[sample {sample.sample_index}] key={sample.sample_key}")
    print(f"current_input: {sample.current_input}")
    print(f"model_response: {sample.model_response}")


def evaluate_task(
    task_config: TaskConfig,
    model_name: str,
    judges: Sequence[JudgeSpec],
    debug: bool,
) -> Dict[str, Any]:
    checkpoint_path = resolve_latest_checkpoint(task_config, model_name)
    npz_payload = load_npz_dicts(checkpoint_path)
    parsed_json_samples = load_task_json_samples(task_config.task_name)
    aligned_samples = align_samples(task_config.task_name, parsed_json_samples, npz_payload)
    sample_scores_path, summary_path, progress_path = get_progress_paths(
        task_config.task_name,
        model_name,
        checkpoint_path,
    )
    output_dir = sample_scores_path.parent
    if debug:
        aligned_samples = aligned_samples[:2]
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    judge_ids = [judge.judge_id for judge in judges]
    diagnostic_mode = len(judges) != 3
    flush_every = 5

    print(f"\n=== Task: {task_config.task_name} ===")
    print(f"checkpoint: {checkpoint_path}")
    print("judges:", ", ".join(f"{judge.judge_id}={judge.model_name}" for judge in judges))
    if diagnostic_mode:
        print("diagnostic_mode: true (judge count is not 3, final aggregation and alpha are disabled)")

    sample_outputs: List[Dict[str, Any]] = []
    sample_output_by_key: Dict[str, Dict[str, Any]] = {}
    completed_keys = set()
    if not debug and sample_scores_path.exists():
        sample_outputs = load_jsonl(sample_scores_path)
        sample_output_by_key = {str(item["sample_key"]): item for item in sample_outputs}
        completed_keys = {
            str(item["sample_key"])
            for item in sample_outputs
            if is_sample_complete(item, judge_ids)
        }
        if sample_outputs:
            print(
                f"resume_from_progress: loaded {len(sample_outputs)} completed samples "
                f"from {sample_scores_path}"
            )
            if progress_path.exists():
                progress = load_progress(progress_path)
                if progress:
                    print(f"progress_state: {progress}")

    completed_since_flush = 0
    for sample in aligned_samples:
        if sample.sample_key in completed_keys:
            continue
        print_sample_header(sample)
        prompt = build_task_prompt(task_config, sample)
        existing_output = sample_output_by_key.get(sample.sample_key)
        if existing_output is None:
            sample_output = make_sample_output(sample)
            sample_outputs.append(sample_output)
            sample_output_by_key[sample.sample_key] = sample_output
        else:
            sample_output = existing_output
        for judge in judges:
            if judge.judge_id in sample_output["parsed_scores"]:
                print(f"  {judge.judge_id}: {sample_output['parsed_scores'][judge.judge_id]} (cached)")
                continue
            try:
                raw_output, parsed_scores = call_judge_model(prompt, judge, task_config.dimensions)
            except Exception:
                if not debug:
                    persist_progress_snapshot(
                        sample_scores_path,
                        progress_path,
                        task_config.task_name,
                        model_name,
                        checkpoint_path,
                        judges,
                        sample_outputs,
                    )
                raise
            sample_output["judge_models"][judge.judge_id] = judge.model_name
            sample_output["raw_judge_outputs"][judge.judge_id] = raw_output
            sample_output["parsed_scores"][judge.judge_id] = parsed_scores
            print(f"  {judge.judge_id}: {parsed_scores}")
        if is_sample_complete(sample_output, judge_ids):
            completed_keys.add(sample.sample_key)
            completed_since_flush += 1
        if not debug and completed_since_flush >= flush_every:
            persist_progress_snapshot(
                sample_scores_path,
                progress_path,
                task_config.task_name,
                model_name,
                checkpoint_path,
                judges,
                sample_outputs,
            )
            completed_since_flush = 0

    dimension_means: Optional[Dict[str, float]] = None
    alpha_by_dimension: Optional[Dict[str, float]] = None
    alpha_mean: Optional[float] = None
    full_agreement_by_dimension: Optional[Dict[str, float]] = None
    pairwise_agreement_by_dimension: Optional[Dict[str, float]] = None
    full_agreement_mean: Optional[float] = None
    pairwise_agreement_mean: Optional[float] = None
    randolph_kappa_by_dimension: Optional[Dict[str, float]] = None
    randolph_kappa_mean: Optional[float] = None
    incomplete_samples = [
        item for item in sample_outputs if not is_sample_complete(item, judge_ids)
    ]
    if incomplete_samples:
        raise RuntimeError(
            f"Found incomplete sample outputs before aggregation: "
            f"{[item['sample_key'] for item in incomplete_samples[:5]]}"
        )
    if not diagnostic_mode:
        _, dimension_means = compute_aggregated_scores(sample_outputs, task_config.dimensions)
        alpha_by_dimension, alpha_mean = compute_alpha(sample_outputs, judge_ids, task_config.dimensions)
        (
            full_agreement_by_dimension,
            pairwise_agreement_by_dimension,
            full_agreement_mean,
            pairwise_agreement_mean,
            randolph_kappa_by_dimension,
            randolph_kappa_mean,
        ) = compute_agreement_metrics(sample_outputs, judge_ids, task_config.dimensions)
        for sample_output in sample_outputs:
            print(
                f"  aggregated[{sample_output['sample_index']}]: "
                f"{sample_output['aggregated_scores']}"
            )
        print(f"dimension_means: {dimension_means}")
        print(f"full_agreement_by_dimension: {full_agreement_by_dimension}")
        print(f"pairwise_agreement_by_dimension: {pairwise_agreement_by_dimension}")
        print(f"full_agreement_mean: {full_agreement_mean}")
        print(f"pairwise_agreement_mean: {pairwise_agreement_mean}")
        print(f"randolph_kappa_by_dimension: {randolph_kappa_by_dimension}")
        print(f"randolph_kappa_mean: {randolph_kappa_mean}")
        print(f"alpha_by_dimension: {alpha_by_dimension}")
        print(f"alpha_mean: {alpha_mean}")

    summary = {
        "task": task_config.task_name,
        "model": model_name,
        "checkpoint_path": str(checkpoint_path),
        "num_samples": len(sample_outputs),
        "judge_models": {judge.judge_id: judge.model_name for judge in judges},
        "diagnostic_mode": diagnostic_mode,
        "dimension_means": dimension_means,
        "full_agreement_by_dimension": full_agreement_by_dimension,
        "pairwise_agreement_by_dimension": pairwise_agreement_by_dimension,
        "full_agreement_mean": full_agreement_mean,
        "pairwise_agreement_mean": pairwise_agreement_mean,
        "randolph_kappa_by_dimension": randolph_kappa_by_dimension,
        "randolph_kappa_mean": randolph_kappa_mean,
        "alpha_by_dimension": alpha_by_dimension,
        "alpha_mean": alpha_mean,
        "timestamp": datetime.now().isoformat(),
    }

    if not debug:
        persist_progress_snapshot(
            sample_scores_path,
            progress_path,
            task_config.task_name,
            model_name,
            checkpoint_path,
            judges,
            sample_outputs,
        )
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        if progress_path.exists():
            progress_path.unlink()

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ESC/ERG response quality with three LLM judges.")
    parser.add_argument("--task", choices=("erg", "esc", "all"), required=True)
    parser.add_argument("--provider", choices=("all", "openai", "mimo", "deepseek"), required=True)
    parser.add_argument(
        "--judge_llm",
        choices=("all", "openai", "mimo", "deepseek"),
        required=True,
    )
    parser.add_argument("--model", required=True, help="Model directory name under results-*.")
    parser.add_argument("--debug", action="store_true", help="Use only 2 samples per dataset and do not write files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not ENV_PATH.exists():
        raise FileNotFoundError(f"Missing env file: {ENV_PATH}")

    judges = build_judge_specs(ENV_PATH, args.provider, args.judge_llm)
    task_names = ["erg", "esc"] if args.task == "all" else [args.task]

    all_summaries = []
    for task_name in task_names:
        summary = evaluate_task(TASKS[task_name], args.model, judges, args.debug)
        all_summaries.append(summary)

    print("\n=== Finished ===")
    for summary in all_summaries:
        print(
            f"{summary['task']}: samples={summary['num_samples']} "
            f"diagnostic_mode={summary['diagnostic_mode']} "
            f"dimension_means={summary['dimension_means']} "
            f"alpha_mean={summary['alpha_mean']}"
        )
    return 0


if __name__ == "__main__":
    """
    comparison：    
                Qwen3-VL-4B-Instruct       
                gpt-5-mini
                gemini-3.1-pro-preview
                Qwen2.5-VL-7B-Instruct
                MiniCPM-V-2_6
                VidEmo-3B
                VidEmo-7B
                Qwen3.5-9B
                mimo-v2.5
"""
    sys.exit(main())
