# data/dataset.py
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import read_video

from config import Config
from utils.misc import FrameSampler
from data.clipping import ClipTransform


class QBPlayDataset(Dataset):
    def __init__(self, root, training=True):
        self.samples = []
        for idx, cls in enumerate(Config.CLASS_NAMES):
            for v in Path(root, cls).glob("*.mp4"):
                self.samples.append((str(v), idx))

        self.sampler = FrameSampler()
        self.transform = ClipTransform(Config.IMG_SIZE, training)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        video, _, _ = read_video(path, pts_unit="sec")

        indices = self.sampler.uniform_indices(
            Config.NUM_FRAMES, video.shape[0]
        )
        frames = [video[i] for i in indices]

        clip = self.transform(frames)
        return clip, label
