# utils/misc.py
import random
import numpy as np
import torch

class Reproducibility:
    @staticmethod
    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


class FrameSampler:
    @staticmethod
    def uniform_indices(num_frames, video_len):
        if video_len <= num_frames:
            return [min(video_len - 1, i) for i in range(num_frames)]
        bins = np.linspace(0, video_len, num_frames + 1, dtype=int)
        return [
            random.randint(bins[i], max(bins[i], bins[i+1]-1))
            for i in range(num_frames)
        ]
