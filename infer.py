# infer.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from config import Config
from data.dataset import QBPlayDataset
from models.x3d import X3DPlayClassifier


def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)

    # Axis labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix (Validation Set)")

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j],
                    ha="center", va="center")

    fig.tight_layout()
    plt.show()


def run_evaluation():
    device = Config.DEVICE

    # Validation dataset
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
    plot_confusion_matrix(cm, Config.CLASS_NAMES)


if __name__ == "__main__":
    run_evaluation()
