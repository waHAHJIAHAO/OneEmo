import random


def get_msd_prompt() -> str:
    promptpool = [
        '[MSD]<video>Please judge based on the context whether the speaker in this round is being sarcastic, answer with yes or no.',
    ]
    return random.choice(promptpool)
