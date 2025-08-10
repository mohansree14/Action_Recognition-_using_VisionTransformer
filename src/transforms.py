# transforms.py
# Preprocessing & augmentation (frame sampling, resizing, etc.)

import random

def temporal_jitter(frames, jitter_range=0.1):
    n = len(frames)
    shift = random.randint(-int(n*jitter_range), int(n*jitter_range))
    frames = frames[max(0, shift):] + frames[:max(0, shift)]
    return frames

def preprocess(frames):
    # Add more transforms if needed
    return frames
