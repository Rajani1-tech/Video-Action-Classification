# train.py
import torch
from torch.utils.data import DataLoader

from config import Config
from data.dataset import QBPlayDataset
from models.x3d import X3DPlayClassifier
from engine.trainer import Trainer
from engine.evaluator import Evaluator
from utils.misc import Reproducibility

Reproducibility.set_seed(Config.SEED)

train_ds = QBPlayDataset(f"{Config.DATA_ROOT}/train", training=True)
val_ds   = QBPlayDataset(f"{Config.DATA_ROOT}/val", training=False)

train_loader = DataLoader(train_ds, Config.BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, Config.BATCH_SIZE)

model = X3DPlayClassifier(Config.NUM_CLASSES).to(Config.DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)

trainer = Trainer(model, optimizer, Config.DEVICE)
evaluator = Evaluator(model, Config.DEVICE)

best_acc = 0
for epoch in range(Config.EPOCHS):
    trainer.train_one_epoch(train_loader)
    acc = evaluator.evaluate(val_loader)

    print(f"Epoch {epoch+1}: Val Acc = {acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), Config.MODEL_PATH)
