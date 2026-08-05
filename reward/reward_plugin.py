import ast
import csv
import importlib.util
import json
import logging
import os
import re
import string
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import requests
except ImportError:  # pragma: no cover - exercised only when optional dependency is absent.
    requests = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover - exercised only in training env differences.
    pd = None

try:
    import torch
except ImportError:  # pragma: no cover - exercised only in training env differences.
    torch = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - exercised only when optional dependency is absent.
    SentenceTransformer = None

try:
    from swift.rewards import ORM, orms
except ImportError:  # pragma: no cover - enables local sanity checks without swift installed.
    class ORM:  # type: ignore[no-redef]
        def __init__(self, args=None, **kwargs):
            self.args = args

    orms = {}


logger = logging.getLogger(__name__)

THINK_FORMAT_PATTERN = re.compile(r"^\s*<think>(?P<think>[\s\S]*?)</think>(?P<answer>[\s\S]+?)\s*$", re.DOTALL)
THINK_CLOSE_TAG = "</think>"
TAG_STRIP_PATTERN = re.compile(r"<[^>]+>")
MER_NORMALIZE_PATTERN = re.compile(r"[^a-z0-9]+")
LIST_SPLIT_PATTERN = re.compile(r"[,;\n\r\t/]|(?:\s+\band\b\s+)", re.IGNORECASE)
PREFIX_PATTERN = re.compile(
    r"^\s*(?:answer|answers|emotion|emotions|feeling|feelings|the\s+emotion|the\s+emotions)\s*[:：-]?\s*",
    re.IGNORECASE,
)
SENTENCE_BERT_MODEL_ENV = "VIDEMO_SENTENCE_BERT_MODEL"
LENGTH_BUDGETS = {
    "classification": {
        "think": {"soft": 130, "hard": 250},
        "answer": {"soft": 5, "hard": 20},
    },
    "OVMER": {
        "think": {"soft": 180, "hard": 250},
        "answer": {"soft": 18, "hard": 30},
    },
    "ERG": {
        "think": {"soft": 200, "hard": 320},
        "answer": {"soft": 130, "hard": 200},
    },
    "ESC": {
        "think": {"soft": 200, "hard": 320},
        "answer": {"soft": 150, "hard": 300},
    },
}
JUDGE_BASE_URL_ENV = "VIDEMO_JUDGE_BASE_URL"
JUDGE_MODEL_ENV = "VIDEMO_JUDGE_MODEL"
JUDGE_TIMEOUT_ENV = "VIDEMO_JUDGE_TIMEOUT"
JUDGE_MAX_RETRIES_ENV = "VIDEMO_JUDGE_MAX_RETRIES"
JUDGE_TEMPERATURE_ENV = "VIDEMO_JUDGE_TEMPERATURE"
JUDGE_MAX_TOKENS_ENV = "VIDEMO_JUDGE_MAX_TOKENS"
DEFAULT_JUDGE_BASE_URL = "http://10.129.30.1:15555/v1"
DEFAULT_JUDGE_MODEL = "Qwen2.5-7B-Instruct"
DEFAULT_JUDGE_TIMEOUT = 30.0
DEFAULT_JUDGE_MAX_RETRIES = 5
DEFAULT_JUDGE_TEMPERATURE = 0.0
DEFAULT_JUDGE_MAX_TOKENS = 256
JSON_BLOCK_PATTERN = re.compile(r"\{[\s\S]*\}")


def _load_prompt_module():
    module_path = Path(__file__).resolve().parent / "llm_judge_prompts" / "prompt.py"
    spec = importlib.util.spec_from_file_location("oneemo_llm_judge_prompt", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load judge prompt module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PROMPT_MODULE = _load_prompt_module()
build_process_judge_messages = _PROMPT_MODULE.build_process_judge_messages
extract_prompt_visual_facts = _PROMPT_MODULE._extract_visual_facts


def _resolve_emotion_wheel_root() -> Path:
    candidates = [
        Path(__file__).resolve().parent / "emotion_wheel",
        Path("/path/to/your/resource/AffectGPT/emotion_wheel"),
        Path("/path/to/your/resource/OV-MER/emotion_wheel"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Cannot find emotion_wheel resources. Expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def _string_to_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
    except Exception:
        parsed = None

    if isinstance(parsed, (list, tuple, set)):
        return [str(item).strip() for item in parsed if str(item).strip()]

    text = text.strip("[](){}")
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def _extract_think_and_answer(text: Any) -> Tuple[str, str]:
    if not isinstance(text, str):
        return "", ""
    close_idx = text.find(THINK_CLOSE_TAG)
    if close_idx == -1:
        return "", text.strip()
    think = text[:close_idx]
    if think.lstrip().startswith("<think>"):
        think = think.split("<think>", 1)[1]
    answer = text[close_idx + len(THINK_CLOSE_TAG):].strip()
    return think.strip(), answer


def _has_single_think_block(text: Any) -> bool:
    if not isinstance(text, str):
        return False
    match = THINK_FORMAT_PATTERN.fullmatch(text)
    if match is None:
        return False

    think = match.group("think")
    answer = match.group("answer")
    if "<think>" in think or "</think>" in think:
        return False
    if "<think>" in answer or "</think>" in answer:
        return False
    return True


def _has_valid_think_format(text: Any) -> bool:
    return _has_single_think_block(text)


def _is_invalid_or_truncated(text: Any) -> bool:
    return not _has_valid_think_format(text)


def _count_words(text: Any) -> int:
    if not isinstance(text, str):
        return 0
    return len(text.split())


def _linear_gate(value: int, soft_limit: int, hard_limit: int) -> float:
    if value <= soft_limit:
        return 1.0
    if value >= hard_limit:
        return 0.0
    if hard_limit <= soft_limit:
        return 0.0
    return (hard_limit - value) / (hard_limit - soft_limit)


def _get_length_budget(task_type: str) -> Optional[Dict[str, Dict[str, int]]]:
    task_type = (task_type or "").upper()
    if task_type in {"MER", "MSA", "MHD", "MSD", "MIR"}:
        return LENGTH_BUDGETS["classification"]
    return LENGTH_BUDGETS.get(task_type)


def _extract_reward_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    _, answer = _extract_think_and_answer(text)
    return answer or text.strip()


def _normalize_mer_label(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    text = TAG_STRIP_PATTERN.sub(" ", text)
    text = text.lower().strip()
    return MER_NORMALIZE_PATTERN.sub("", text)


def _normalize_classification_label(text: Any) -> str:
    return _normalize_mer_label(text)


def _normalize_ovmer_label(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    text = TAG_STRIP_PATTERN.sub(" ", text)
    text = text.replace("_", " ").lower()
    text = PREFIX_PATTERN.sub("", text)
    text = re.sub(rf"^[{re.escape(string.punctuation)}\s]+", "", text)
    text = re.sub(rf"[{re.escape(string.punctuation)}\s]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _parse_ovmer_labels(text: Any) -> List[str]:
    if not isinstance(text, str):
        return []

    _, answer = _extract_think_and_answer(text)
    answer = answer or str(text).strip()
    answer = PREFIX_PATTERN.sub("", answer)
    answer = answer.replace("，", ",").replace("；", ";").replace("、", ",")

    candidates = _string_to_list(answer)
    if not candidates:
        candidates = LIST_SPLIT_PATTERN.split(answer)

    labels: List[str] = []
    seen = set()
    for item in candidates:
        norm = _normalize_ovmer_label(item)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        labels.append(norm)
    return labels


def _ensure_list(value: Any, batch_size: int) -> List[Any]:
    if isinstance(value, list):
        if len(value) == batch_size:
            return value
        if len(value) == 1 and batch_size > 1:
            return value * batch_size
        return value[:batch_size]
    if batch_size <= 0:
        return []
    return [value for _ in range(batch_size)]


def _extract_assistant_content_from_message_list(message_list: Any) -> str:
    if not isinstance(message_list, list):
        return ""
    for item in reversed(message_list):
        if isinstance(item, dict) and item.get("role") == "assistant":
            return str(item.get("content", "")).strip()
    return ""


def _maybe_get_prompt_content(message_list: Any) -> str:
    if not isinstance(message_list, list):
        return ""
    for item in message_list:
        if isinstance(item, dict) and item.get("role") == "user":
            return str(item.get("content", "")).strip()
    return ""


def _normalize_messages_batch(messages: Any, batch_size: int) -> Optional[List[Any]]:
    if not isinstance(messages, list):
        return None
    if messages and isinstance(messages[0], dict):
        return [messages]
    return _ensure_list(messages, batch_size)


def _resolve_reference_texts(batch_size: int, solution: Any = None, **kwargs) -> List[str]:
    if solution is not None:
        values = _ensure_list(solution, batch_size)
        return [str(item).strip() if item is not None else "" for item in values]

    messages = _normalize_messages_batch(kwargs.get("messages"), batch_size)
    if messages is not None:
        return [_extract_assistant_content_from_message_list(item) for item in messages]

    return ["" for _ in range(batch_size)]


def _resolve_reasoning_refs(batch_size: int, **kwargs) -> List[str]:
    if "reasoning_ref" in kwargs:
        values = _ensure_list(kwargs["reasoning_ref"], batch_size)
        return [str(item).strip() if item is not None else "" for item in values]
    return ["" for _ in range(batch_size)]


def _resolve_perceptions(batch_size: int, **kwargs) -> List[Any]:
    if "perception" in kwargs:
        return _ensure_list(kwargs["perception"], batch_size)
    return [None for _ in range(batch_size)]


def _resolve_messages(batch_size: int, **kwargs) -> List[Any]:
    messages = _normalize_messages_batch(kwargs.get("messages"), batch_size)
    if messages is not None:
        return messages
    return [[] for _ in range(batch_size)]


def _resolve_task_types(batch_size: int, **kwargs) -> List[str]:
    if "task_type" in kwargs:
        values = _ensure_list(kwargs["task_type"], batch_size)
        return [str(item).strip().upper() if item is not None else "" for item in values]

    messages = _normalize_messages_batch(kwargs.get("messages"), batch_size)
    if messages is not None:
        values = messages
        task_types: List[str] = []
        for item in values:
            prompt = _maybe_get_prompt_content(item)
            match = re.match(r"\s*\[([A-Za-z0-9_]+)\]", prompt)
            task_types.append(match.group(1).upper() if match else "")
        return task_types

    return ["" for _ in range(batch_size)]


def _safe_int_from_env(env_name: str, default: int) -> int:
    raw = os.environ.get(env_name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid int env %s=%r, falling back to %s", env_name, raw, default)
        return default


def _safe_float_from_env(env_name: str, default: float) -> float:
    raw = os.environ.get(env_name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float env %s=%r, falling back to %s", env_name, raw, default)
        return default


def _get_judge_base_url() -> str:
    base_url = os.environ.get(JUDGE_BASE_URL_ENV, DEFAULT_JUDGE_BASE_URL).strip()
    return base_url.rstrip("/")


def _get_judge_model() -> str:
    model = os.environ.get(JUDGE_MODEL_ENV, DEFAULT_JUDGE_MODEL).strip()
    return model or DEFAULT_JUDGE_MODEL


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


def _extract_json_payload(text: str) -> Dict[str, Any]:
    stripped = _strip_markdown_fences(text)
    if not stripped:
        raise ValueError("Judge response is empty.")

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = JSON_BLOCK_PATTERN.search(stripped)
    if match is None:
        raise ValueError("Judge response does not contain a JSON object.")

    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("Judge response JSON root must be an object.")
    return payload


def _normalize_subscore(payload: Dict[str, Any], key: str) -> int:
    if key not in payload:
        raise ValueError(f"Missing judge score field: {key}")
    value = payload[key]
    if isinstance(value, bool):
        raise ValueError(f"Judge field {key} must be an integer score, got bool.")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"Judge field {key} must be an integer score, got float {value}.")
    try:
        score = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Judge field {key} is not an integer score: {value!r}") from exc
    if score < 0 or score > 5:
        raise ValueError(f"Judge field {key} is out of range [0, 5]: {score}")
    return score


def _validate_process_payload(task_type: str, payload: Dict[str, Any]) -> float:
    task_type = (task_type or "").upper()
    if task_type in {"ERG", "ESC"}:
        score1 = _normalize_subscore(payload, "user_state_alignment")
        score2 = _normalize_subscore(payload, "response_strategy_alignment")
    else:
        score1 = _normalize_subscore(payload, "fact_consistency")
        score2 = _normalize_subscore(payload, "reasoning_answer_coherence")
    return (score1 + score2) / 10.0


def _judge_chat_completion(messages: List[Dict[str, str]], task_type: str) -> float:
    if requests is None:
        raise ImportError("requests is required for ProcessReward judge calls.")

    base_url = _get_judge_base_url()
    model = _get_judge_model()
    timeout = _safe_float_from_env(JUDGE_TIMEOUT_ENV, DEFAULT_JUDGE_TIMEOUT)
    max_retries = _safe_int_from_env(JUDGE_MAX_RETRIES_ENV, DEFAULT_JUDGE_MAX_RETRIES)
    temperature = _safe_float_from_env(JUDGE_TEMPERATURE_ENV, DEFAULT_JUDGE_TEMPERATURE)
    max_tokens = _safe_int_from_env(JUDGE_MAX_TOKENS_ENV, DEFAULT_JUDGE_MAX_TOKENS)

    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_error = "unknown error"
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            response_payload = response.json()
            content = response_payload["choices"][0]["message"]["content"]
            judge_payload = _extract_json_payload(str(content))
            return _validate_process_payload(task_type, judge_payload)
        except Exception as exc:
            response_text = ""
            if "response" in locals():
                try:
                    response_text = response.text[:400]
                except Exception:
                    response_text = ""
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Judge request failed for task_type=%s attempt=%d/%d error=%s response=%r",
                task_type,
                attempt,
                max_retries,
                last_error,
                response_text,
            )

    raise RuntimeError(f"Judge request failed after {max_retries} retries: {last_error}")


@lru_cache(maxsize=1)
def _load_sentence_bert_model() -> Any:
    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers is required for ERG/ESC answer reward. "
            "Please install it in the OneEmo environment."
        )

    model_path = os.environ.get(SENTENCE_BERT_MODEL_ENV)
    if not model_path:
        raise ImportError(
            f"{SENTENCE_BERT_MODEL_ENV} is not set. "
            "Please export the local Sentence-BERT model path before training."
        )

    resolved_path = Path(model_path).expanduser()
    if not resolved_path.exists():
        raise ImportError(
            f"Sentence-BERT model path does not exist: {resolved_path}"
        )

    device = "cuda:0" if torch is not None and torch.cuda.is_available() else "cpu"
    model_kwargs = {"torch_dtype": torch.float16} if device.startswith("cuda") and torch is not None else None
    logger.info("Loading Sentence-BERT from %s on device=%s", resolved_path, device)
    return SentenceTransformer(
        str(resolved_path),
        device=device,
        model_kwargs=model_kwargs,
    )


def _compute_text_cosine_reward(pred_text: str, ref_text: str) -> Optional[float]:
    if not ref_text:
        return None
    if not pred_text:
        return 0.0

    model = _load_sentence_bert_model()
    embeddings = model.encode(
        [pred_text, ref_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    pred_embedding, ref_embedding = embeddings
    cosine_sim = float(pred_embedding @ ref_embedding)
    return max(cosine_sim, 0.0)


def _read_format_mapping(root: Path) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    format_path = root / "format.csv"
    with format_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw = str(row.get("name", "")).strip().lower()
            formats = _string_to_list(row.get("format", ""))
            for fmt in formats:
                key = fmt.strip().lower()
                if not key:
                    continue
                mapping.setdefault(key, []).append(raw)
            if raw:
                mapping.setdefault(raw, []).append(raw)
    return mapping


def _read_wheel_to_map(xlsx_path: Path) -> Dict[str, Dict[str, List[str]]]:
    if pd is None:
        raise ImportError("pandas is required to read emotion wheel excel files.")
    dataframe = pd.read_excel(xlsx_path)
    store_map: Dict[str, Dict[str, List[str]]] = {}
    level1 = level2 = level3 = ""
    for _, row in dataframe.iterrows():
        if not pd.isna(row["level1"]):
            level1 = str(row["level1"]).lower().strip()
        if not pd.isna(row["level2"]):
            level2 = str(row["level2"]).lower().strip()
        if not pd.isna(row["level3"]):
            level3 = str(row["level3"]).lower().strip()
        store_map.setdefault(level1, {}).setdefault(level2, []).append(level3)
    return store_map


def _convert_all_wheels_to_candidate_labels(root: Path) -> List[str]:
    labels = set()
    for wheel_path in sorted(root.glob("wheel*.xlsx")):
        wheel_map = _read_wheel_to_map(wheel_path)
        for level1, level2_map in wheel_map.items():
            labels.add(level1)
            for level2, level3_list in level2_map.items():
                labels.add(level2)
                labels.update(level3_list)
    return sorted(labels)


def _merge_mapping(map1: Dict[str, List[str]], map2: Dict[str, List[str]]) -> Dict[str, List[str]]:
    merged = {key: list(values) for key, values in map1.items()}
    for key, values in map2.items():
        merged.setdefault(key, [])
        merged[key] = sorted(set(merged[key] + values))
    return merged


def _read_candidate_synonym_merge(root: Path) -> Dict[str, List[str]]:
    if pd is None:
        raise ImportError("pandas is required to read emotion wheel synonym files.")
    dataframe = pd.read_excel(root / "synonym.xlsx")
    wheel_labels = set(_convert_all_wheels_to_candidate_labels(root))
    merged: Dict[str, List[str]] = {}

    for run_idx in range(1, 9):
        run_mapping: Dict[str, List[str]] = {}
        word_col = f"word_run{run_idx}"
        synonym_col = f"synonym_run{run_idx}"
        for _, row in dataframe.iterrows():
            raw = str(row[word_col]).strip().lower()
            if raw not in wheel_labels:
                continue
            run_mapping.setdefault(raw, []).append(raw)
            for synonym in _string_to_list(row[synonym_col]):
                key = synonym.strip().lower()
                if not key:
                    continue
                run_mapping.setdefault(key, []).append(raw)
        merged = _merge_mapping(merged, run_mapping)
    return merged


def _get_wheel_cluster(root: Path, wheel: str, level: str = "level1") -> Dict[str, str]:
    emotion_wheel = _read_wheel_to_map(root / f"{wheel}.xlsx")
    wheel_map: Dict[str, str] = {}

    if level == "level1":
        for level1, level2_map in emotion_wheel.items():
            wheel_map[level1] = level1
            for level2, level3_list in level2_map.items():
                wheel_map[level2] = level1
                for level3 in level3_list:
                    wheel_map[level3] = level1
        return wheel_map

    raise ValueError(f"Unsupported wheel level: {level}")


@lru_cache(maxsize=1)
def _load_ovmer_resources() -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, Dict[str, str]]]:
    root = _resolve_emotion_wheel_root()
    format_mapping = _read_format_mapping(root)
    raw_mapping = _read_candidate_synonym_merge(root)
    wheel_maps = {
        f"wheel{idx}": _get_wheel_cluster(root, f"wheel{idx}", "level1")
        for idx in range(1, 6)
    }
    return format_mapping, raw_mapping, wheel_maps


def _map_labels_to_wheel(
    labels: Sequence[str],
    wheel_map: Dict[str, str],
    format_mapping: Dict[str, List[str]],
    raw_mapping: Dict[str, List[str]],
) -> List[str]:
    mapped: List[str] = []
    for label in labels:
        if label not in format_mapping:
            continue
        candidates = []
        for fmt in format_mapping[label]:
            candidates.extend(raw_mapping.get(fmt, []))
        wheel_labels = sorted({wheel_map[item] for item in candidates if item in wheel_map})
        if wheel_labels:
            mapped.append(wheel_labels[0])
    return mapped


def _f1_score(pred_items: Iterable[str], gt_items: Iterable[str]) -> float:
    pred_set = set(pred_items)
    gt_set = set(gt_items)
    if not pred_set and not gt_set:
        return 1.0
    if not pred_set or not gt_set:
        return 0.0
    intersection = len(pred_set & gt_set)
    precision = intersection / len(pred_set)
    recall = intersection / len(gt_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_ovmer_waf(pred_labels: Sequence[str], gt_labels: Sequence[str]) -> Optional[float]:
    if not pred_labels and not gt_labels:
        return 1.0
    if not gt_labels:
        return None

    format_mapping, raw_mapping, wheel_maps = _load_ovmer_resources()
    scores = []
    pred_norm = [_normalize_ovmer_label(item) for item in pred_labels if _normalize_ovmer_label(item)]
    gt_norm = [_normalize_ovmer_label(item) for item in gt_labels if _normalize_ovmer_label(item)]
    if not gt_norm:
        return None

    for wheel_name in ("wheel1", "wheel2", "wheel3", "wheel4", "wheel5"):
        wheel_map = wheel_maps[wheel_name]
        pred_mapped = _map_labels_to_wheel(pred_norm, wheel_map, format_mapping, raw_mapping)
        gt_mapped = _map_labels_to_wheel(gt_norm, wheel_map, format_mapping, raw_mapping)
        scores.append(_f1_score(pred_mapped, gt_mapped))
    return sum(scores) / len(scores)


class FormatReward(ORM):
    def __call__(self, completions, **kwargs) -> List[float]:
        return [
            1.0 if _has_valid_think_format(completion) else 0.0
            for completion in completions
        ]

class ProcessReward(ORM):
    CLASSIFICATION_TASK_TYPES = {"MER", "MSA", "MHD", "MSD", "MIR", "MUD"}
    OV_CLASSIFICATION_TASK_TYPES = {"OVMER"}
    INTERACTION_TASK_TYPES = {"ERG", "ESC"}

    def __call__(self, completions, **kwargs) -> List[float]:
        batch_size = len(completions)
        task_types = _resolve_task_types(batch_size, **kwargs)
        reasoning_refs = _resolve_reasoning_refs(batch_size, **kwargs)
        perceptions = _resolve_perceptions(batch_size, **kwargs)
        messages_batch = _resolve_messages(batch_size, **kwargs)
        return [
            self._score_one(
                completion=completion,
                task_type=task_type,
                messages=messages,
                perception=perception,
                reasoning_ref=reasoning_ref,
            )
            for completion, task_type, messages, perception, reasoning_ref in zip(
                completions, task_types, messages_batch, perceptions, reasoning_refs
            )
        ]

    def _score_one(
        self,
        completion: Any,
        task_type: str,
        messages: Any,
        perception: Any,
        reasoning_ref: str,
    ) -> float:
        if not isinstance(completion, str):
            return 0.0
        if _is_invalid_or_truncated(completion):
            return 0.0

        task_type = (task_type or "").upper()
        if task_type in self.INTERACTION_TASK_TYPES:
            ref_think, _ = _extract_think_and_answer(reasoning_ref)
            if not ref_think:
                logger.warning("Missing usable reasoning_ref think section for task_type=%s", task_type)
                return 0.0
        elif task_type in self.CLASSIFICATION_TASK_TYPES or task_type in self.OV_CLASSIFICATION_TASK_TYPES:
            visual_facts = extract_prompt_visual_facts(perception)
            if not visual_facts:
                logger.warning("Missing usable perception.visual_facts for task_type=%s", task_type)
                return 0.0
        else:
            logger.warning("Unsupported task_type for ProcessReward: %s", task_type)
            return 0.0

        try:
            judge_messages = build_process_judge_messages(
                task_type=task_type,
                messages=messages,
                completion=completion,
                perception=perception,
                reasoning_ref=reasoning_ref,
            )
            return _judge_chat_completion(judge_messages, task_type)
        except Exception as exc:
            logger.warning("ProcessReward fallback to 0 for task_type=%s error=%s", task_type, exc)
            return 0.0

class AnswerReward(ORM):
    CLASSIFICATION_TASK_TYPES = {"MER", "MSA", "MHD", "MSD", "MIR"}
    OV_CLASSIFICATION_TASK_TYPES = {"OVMER"}
    INTERACTION_TASK_TYPES = {"ERG", "ESC"}

    def __call__(self, completions, solution=None, **kwargs) -> List[Optional[float]]:
        batch_size = len(completions)
        task_types = _resolve_task_types(batch_size, **kwargs)
        references = _resolve_reference_texts(batch_size, solution=solution, **kwargs)
        return [
            self._score_one(completion, task_type, reference)
            for completion, task_type, reference in zip(completions, task_types, references)
        ]

    def _score_one(self, completion: Any, task_type: str, reference: str) -> Optional[float]:
        if not isinstance(completion, str):
            return 0.0
        if not reference:
            logger.warning("Missing reference answer for task_type=%s; skipping reward.", task_type)
            return None
        if _is_invalid_or_truncated(completion):
            return 0.0

        pred_answer = _extract_reward_text(completion)
        gt_answer = _extract_reward_text(reference)

        task_type = (task_type or "").upper()
        base_score: Optional[float]
        if task_type in self.CLASSIFICATION_TASK_TYPES:
            pred_label = _normalize_classification_label(pred_answer)
            gt_label = _normalize_classification_label(gt_answer)
            base_score = 1.0 if pred_label == gt_label else 0.0
        elif task_type in self.OV_CLASSIFICATION_TASK_TYPES:
            base_score = compute_ovmer_waf(_parse_ovmer_labels(pred_answer), _parse_ovmer_labels(gt_answer))
        elif task_type in self.INTERACTION_TASK_TYPES:
            base_score = _compute_text_cosine_reward(pred_answer, gt_answer)
            if base_score is None:
                logger.warning("Missing reference answer for task_type=%s; skipping reward.", task_type)
        else:
            return None

        if base_score is None:
            return None

        budget = _get_length_budget(task_type)
        if budget is None:
            return base_score

        think_text, answer_text = _extract_think_and_answer(completion)
        think_gate = _linear_gate(
            _count_words(think_text),
            budget["think"]["soft"],
            budget["think"]["hard"],
        )
        answer_gate = _linear_gate(
            _count_words(answer_text),
            budget["answer"]["soft"],
            budget["answer"]["hard"],
        )
        return base_score * min(think_gate, answer_gate)


orms["format"] = FormatReward
orms["process"] = ProcessReward
orms["answer"] = AnswerReward
