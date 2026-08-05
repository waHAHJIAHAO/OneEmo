import random

def get_mer_prompt() -> str:
    promptpool = [
        "[MER]<video>Look at the video and identify the character’s main emotion. Answer with a single emotion.",
        "[MER]<video>Determine the character's primary emotion, Answer with a single emotion.",
        "[MER]<video>Please identify the emotions of the person in the video. Answer with a single emotion.",
        "[MER]<video>What emotion best describes the character’s state in the video?",  
    ]
    return promptpool[2]