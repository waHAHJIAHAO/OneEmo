import random

def get_msa_prompt() -> str:
    promptpool = [
        "[MSA]<video>Please analyze the sentiment of the characters in the video and label them as positive, neutral or negative.",
    ]
    return random.choice(promptpool)
