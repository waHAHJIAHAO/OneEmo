import random

def get_mir_prompt(candidate_labels: str) -> str:
    promptpool = [
        f"[MIR]<video>Please recognize speaker's intention from the provided candidate labels: {candidate_labels}.",
        f"[MIR]<video>Based on what the speaker says and how they act, what is their main goal or intention? Choose the best match: {candidate_labels}.",
        f"[MIR]<video>From the video, what is the speaker trying to achieve or express? Pick the intention that fits best: {candidate_labels}.",
        f"[MIR]<video>Considering the speaker's dialogue, expressions, and actions, which intention does their message convey: {candidate_labels}.",
    ]
    return promptpool[0]
