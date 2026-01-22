# data/transforms.py
import torch
import numpy as np
import torchvision.transforms.functional as TF

class ClipTransform:
    def __init__(self, img_size: int, training: bool):
        self.img_size = img_size
        self.training = training

    def _to_pil(self, frame):
        arr = frame.numpy() if isinstance(frame, torch.Tensor) else frame
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        if arr.shape[-1] > 3:
            arr = arr[..., :3]
        return TF.to_pil_image(arr.astype(np.uint8))

    def __call__(self, frames):
        frames = [self._to_pil(f) for f in frames]
        frames = [TF.resize(f, (256,256)) for f in frames]
        frames = [TF.center_crop(f, (self.img_size, self.img_size)) for f in frames]

        if self.training and np.random.rand() < 0.5:
            frames = [TF.hflip(f) for f in frames]

        clip = torch.stack([TF.to_tensor(f) for f in frames], dim=1)

        mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1,1)
        std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1,1)

        return (clip - mean) / std
