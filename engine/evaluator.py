# engine/evaluator.py
import torch

class Evaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def evaluate(self, loader):
        self.model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for clips, labels in loader:
                clips, labels = clips.to(self.device), labels.to(self.device)
                preds = self.model(clips).argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return correct / total
