"""Prompt builders for rubric-based LLM judges used by process rewards."""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

CLASSIFICATION_TASK_TYPES = {"MSA", "MER", "OVMER", "MIR", "MSD", "MHD", "MUD"}
INTERACTION_TASK_TYPES = {"ERG", "ESC"}

JUDGE_OUTPUT_SCHEMA = """
Return valid JSON only. Do not wrap it in markdown fences.

{
  "fact_consistency": 0-5 integer score,
  "reasoning_answer_coherence": 0-5 integer score
}
""".strip()

INTERACTION_OUTPUT_SCHEMA = """
Return valid JSON only. Do not wrap it in markdown fences.

{
  "user_state_alignment": 0-5 integer score,
  "response_strategy_alignment": 0-5 integer score
}
""".strip()

CLASSIFICATION_JUDGE_SYSTEM_PROMPT = """
You are a rigorous evaluator of multimodal reasoning processes and reward judgments.  
Your task is to assess a candidate's performance on tasks such as sentiment analysis, emotion recognition, intent detection, sarcasm detection, humor understanding, and related multimodal classification—based on their extraction and comprehension of atomic-level behavioral descriptions from visual content—and to evaluate the internal consistency between their reasoning process and final answer. Before assessment, you must first separate and extract two types of content from the candidate’s reasoning text: (1) atomic-level behavioral descriptions based on visual cues (e.g., facial expressions, body movements, scene elements—objective observations), and (2) inferences and judgments drawn from these behavioral descriptions. Evaluation will be conducted based on the extracted behaviors and reasoning content, not directly on the original text.

Precisely assess the following two dimensions:

1. Visual Fact Consistency:  
Check whether the model’s internal reasoning aligns with the provided atomic visual facts and task content.  
Reward explanations that do not conflict with the given facts; penalize fabricated observations, invented events, made-up emotions, or assertions that directly contradict the evidence.  
Allow reasonable high-level inferences when they are not contradicted by observable cues or explicit text.  
Do not require every inference to reference each fact; focus on whether key conclusions are supported by evidence or at least do not contradict it.

- Factual Consistency Scoring Criteria:  
5 points: No claims clearly contradict the provided facts or explicit text; no substantial hallucinations.  
4 points: Essentially consistent. Minor unsupported details present but do not affect core judgment.  
3 points: Partially consistent. Some important inferences are weakly grounded, but overall logic remains acceptable.  
2 points: Multiple significant claims lack basis, are exaggerated, or involve selective interpretation of evidence.  
1 point: Most reasoning relies on fictional cues or clearly conflicts with evidence.  
0 points: No usable basis, missing or irrelevant reasoning, or reasoning explicitly contradicted by evidence.

2. Evaluate whether the candidate’s reasoning content is internally consistent with their final answer—that is, whether the reasoning genuinely supports the final answer—rather than assessing the correctness of the answer itself.  
The answer should naturally follow from the reasoning.  
For single-label tasks, reasoning should converge to a clear label or a definitive "yes/no" judgment.  
For open-label tasks (e.g., OVMER), where no predefined label set exists, simply determine whether the reasoning is internally consistent with the predicted label set—i.e., whether the reasoning supports that specific set rather than an alternative one.  
This part of the reward focuses solely on the coherence of the reasoning that leads to the answer, and is not affected by the truthfulness of the previous facts; only judge internal consistency between reasoning and final answer. 

- Rreasoning answer coherence Scoring Criteria:
5 points: Reasoning clearly and directly supports the final answer, with no contradictions or unresolved alternatives.  
4 points: Essentially consistent. Answer matches reasoning, with only minor ambiguity or brevity.  
3 points: Roughly compatible, but the logical chain from reasoning to answer is incomplete, vague, or omits key clarifying steps.  
2 points: Clear mismatch. Some reasoning points in a different direction, or the final answer lacks sufficient support.  
1 point: Severe contradiction between reasoning and final answer.  
0 points: Missing reasoning or answer, or answer cannot be logically inferred from reasoning.

**General Instructions:**  
Use only the information provided in the prompt.  
Ignore fluency, verbosity, and formatting quality unless they affect judgment.  
The VLM provides atomic-level visual behavior descriptions, not emotional labels. As long as they do not contradict the provided facts, inferring internal states from visual cues is permitted in factual consistency evaluation.

Output format:
""" + JUDGE_OUTPUT_SCHEMA

INTERACTION_JUDGE_SYSTEM_PROMPT = """
You are a strict process-reward judge for empathetic or counseling-style reasoning.

Your job is to evaluate the candidate completion for ERG and ESC tasks. Judge the quality of the reasoning process, not the wording style of the final response.

Evaluate exactly two dimensions:

1. user_state_alignment
Compare the candidate reasoning with the reference reasoning at a semantic level.
Focus on whether the candidate correctly captures the user's current situation, emotional state, core conflict, needs, and likely drivers.
Exact wording is not required. Equivalent paraphrases should receive full credit.

Rubric for user_state_alignment:
- 5: Fully aligned. Captures the user's state, emotions, needs, and core conflict with no material drift.
- 4: Mostly aligned. Minor omissions or wording differences, but the core state is captured.
- 3: Partially aligned. Some central aspects are captured, but important emotional or situational details are missing or blurred.
- 2: Weak alignment. Captures only a small part of the user state, or mixes it with notable misunderstanding.
- 1: Largely misaligned. Describes a substantially different user state or emotional meaning.
- 0: No usable user-state analysis, or irrelevant to the reference reasoning.

2. response_strategy_alignment
Compare the strategy implied in the candidate reasoning with the reference reasoning.
Focus on the intended response plan, sequencing, and prioritization: validation, exploration, reassurance, reflective questioning, emotional labeling, gentle reframing, practical support, or other therapeutic moves.

Rubric for response_strategy_alignment:
- 5: Fully aligned. Matches the reference strategy, support goal, and prioritization of moves.
- 4: Mostly aligned. Same broad strategy with only minor omissions or reduced specificity.
- 3: Partially aligned. Relevant but misses an important component of the reference strategy.
- 2: Weak alignment. Only superficial overlap or a noticeably off-target direction.
- 1: Strong mismatch. Substantially different or counterproductive strategy.
- 0: No usable strategy, or irrelevant to the reference reasoning.

General instructions:
- Use semantic matching, not string matching.
- Compare only the reasoning inside the think sections.
- Do not compare or score the final response text.

Output format:
""" + INTERACTION_OUTPUT_SCHEMA


def _safe_load_json_like(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}

    text = value.strip()
    if not text:
        return {}

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    try:
        parsed = ast.literal_eval(text)
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, SyntaxError):
        return {}


def _extract_visual_facts(perception: Any) -> List[str]:
    payload = _safe_load_json_like(perception)
    facts = payload.get("visual_facts", [])
    if not isinstance(facts, list):
        return []
    return [str(item).strip() for item in facts if str(item).strip()]


def _extract_messages_text(messages: Any) -> Tuple[str, str]:
    user_text = ""
    assistant_text = ""
    if not isinstance(messages, list):
        return user_text, assistant_text

    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role == "user" and not user_text:
            user_text = content
        elif role == "assistant":
            assistant_text = content
    return user_text, assistant_text


def _extract_think_and_answer(text: Any) -> Tuple[str, str]:
    if not isinstance(text, str):
        return "", ""
    close_tag = "</think>"
    close_idx = text.find(close_tag)
    if close_idx < 0:
        return "", text.strip()

    think = text[:close_idx]
    if "<think>" in think:
        think = think.split("<think>", 1)[1]
    answer = text[close_idx + len(close_tag):].strip()
    return think.strip(), answer


def _format_numbered_lines(items: Sequence[str], empty_placeholder: str) -> str:
    clean_items = [str(item).strip() for item in items if str(item).strip()]
    if not clean_items:
        return f"1. {empty_placeholder}"
    return "\n".join(f"{idx}. {item}" for idx, item in enumerate(clean_items, start=1))


def _task_family(task_type: Optional[str]) -> str:
    task = str(task_type or "").strip().upper()
    if task in CLASSIFICATION_TASK_TYPES:
        return "classification"
    if task in INTERACTION_TASK_TYPES:
        return "interaction"
    return "unknown"


def build_classification_process_judge_messages(
    *,
    task_type: str,
    messages: Any,
    perception: Any,
    completion: str,
) -> List[Dict[str, str]]:
    user_prompt, _ = _extract_messages_text(messages)
    visual_facts = _extract_visual_facts(perception)
    candidate_think, candidate_answer = _extract_think_and_answer(completion)

    user_content = f"""
Task type: {task_type}

Original task instruction:
{user_prompt or "[missing user instruction]"}

Atomic visual facts from the VLM:
{_format_numbered_lines(visual_facts, "[no visual facts provided]")}

Candidate think section:
{candidate_think or "[missing think section]"}

Candidate final answer:
{candidate_answer or "[missing final answer]"}

Scoring reminder:
- Score only `fact_consistency` and `reasoning_answer_coherence`.
- For OVMER, judge whether the reasoning supports the predicted label set as a whole.
""".strip()

    return [
        {"role": "system", "content": CLASSIFICATION_JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_interaction_process_judge_messages(
    *,
    task_type: str,
    messages: Any,
    completion: str,
    reasoning_ref: Optional[str],
) -> List[Dict[str, str]]:
    user_prompt, _ = _extract_messages_text(messages)
    candidate_think, _ = _extract_think_and_answer(completion)
    ref_think, _ = _extract_think_and_answer(reasoning_ref or "")

    user_content = f"""
Task type: {task_type}

Original task instruction and user context:
{user_prompt or "[missing user instruction]"}

Reference think section:
{ref_think or "[missing reference think section]"}

Candidate think section:
{candidate_think or "[missing think section]"}

Scoring reminder:
- Score only `user_state_alignment` and `response_strategy_alignment`.
- Compare only the semantic content of the two think sections.
- Do not compare final response wording.
""".strip()

    return [
        {"role": "system", "content": INTERACTION_JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_process_judge_messages(
    *,
    task_type: str,
    messages: Any,
    completion: str,
    perception: Any = None,
    reasoning_ref: Optional[str] = None,
) -> List[Dict[str, str]]:
    family = _task_family(task_type)
    if family == "classification":
        return build_classification_process_judge_messages(
            task_type=task_type,
            messages=messages,
            perception=perception,
            completion=completion,
        )
    if family == "interaction":
        return build_interaction_process_judge_messages(
            task_type=task_type,
            messages=messages,
            completion=completion,
            reasoning_ref=reasoning_ref,
        )
    raise ValueError(f"Unsupported task_type for process judge prompt: {task_type}")
