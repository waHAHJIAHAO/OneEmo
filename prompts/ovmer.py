import random

def get_ovmer_prompt() -> str:
    promptpool = [
        "[OVMER]<video>Please recognize all possible emotional states of the character.",
        "[OVMER]<video>Based on the video, describe the range of emotions the character may be feeling.",
        "[OVMER]<video>What feelings does the character show? List them briefly.",
        "[OVMER]<video>Based on the video, describe the emotions of the characters using open-vocabulary emotional words.",
    ]
    return promptpool[2]
