# config.py
import torch

class Config:
    DATA_ROOT = "/home/predator/Desktop/college_project/Video-Action-Classification/dataset"
    MODEL_PATH = "/home/predator/Desktop/college_project/Video-Action-Classification/x3d_sidx.pth"

    NUM_CLASSES = 2
    CLASS_NAMES = ["run-play", "pass-play"]

    NUM_FRAMES = 32
    IMG_SIZE = 224

    BATCH_SIZE = 1
    EPOCHS = 55
    LR = 1e-4
    NUM_WORKERS = 4

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
