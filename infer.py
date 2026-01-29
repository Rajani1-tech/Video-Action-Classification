# infer.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from config import Config
from data.dataset import QBPlayDataset
from models.x3d import X3DPlayClassifier


def plot_confusion_matrix(cm, class_names, save_path='assets/cm.png', title="Confusion Matrix (Validation Set)"):
    plt.figure(figsize=(8, 6))
 
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names) 

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.savefig(save_path)
    print(f"Confusion matrix saved at: {save_path}")
    
    plt.show()

def run_evaluation():
    device = Config.DEVICE

  
    val_dataset = QBPlayDataset(
        root=f"{Config.DATA_ROOT}/val",
        training=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False
    )

    # Load model
    model = X3DPlayClassifier(Config.NUM_CLASSES).to(device)
    model.load_state_dict(
        torch.load(Config.MODEL_PATH, map_location=device)
    )
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for clips, labels in val_loader:
            clips = clips.to(device)
            labels = labels.to(device)

            outputs = model(clips)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:\n", cm)


    print("\nClassification Report:")
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=Config.CLASS_NAMES
        )
    )

    # Plot
    plot_confusion_matrix(cm, Config.CLASS_NAMES, save_path='assets/cm.png')



if __name__ == "__main__":
    run_evaluation()
