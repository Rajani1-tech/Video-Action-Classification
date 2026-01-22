# engine/trainer.py
import torch
import torch.nn as nn

class Trainer:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()

    def train_one_epoch(self, loader):
        self.model.train()
        for clips, labels in loader:
            clips, labels = clips.to(self.device), labels.to(self.device)

            loss = self.criterion(self.model(clips), labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
