import random


def get_mhd_prompt() -> str:
    promptpool = [
        '[MHD]<video>Based on the context of the speaker and visual cues, determine whether there is a humorous expression, answer with yes or no.'
    ]
    return random.choice(promptpool)
