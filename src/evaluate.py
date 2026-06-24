

import os

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix

from dataset import CLASS_NAMES, get_dataloaders
from model import build_model

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.tolist())

    return all_labels, all_preds


def plot_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("CIFAR-10 Confusion Matrix")

    # Annotate each cell with its count
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Confusion matrix saved to {save_path}")
    plt.show()


def main():
    device = get_device()
    print(f"Using device: {device}")

    _, test_loader = get_dataloaders()
    model = build_model().to(device)

    checkpoint_path = os.path.join(PROJECT_ROOT, "src", "checkpoints", "best_model.pt")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded checkpoint from {checkpoint_path}")

    labels, preds = collect_predictions(model, test_loader, device)

    print("\nClassification Report:")
    report = classification_report(labels, preds, target_names=CLASS_NAMES, digits=4)
    print(report)

    cm = confusion_matrix(labels, preds)
    cm_path = os.path.join(PROJECT_ROOT, "src", "outputs", "confusion_matrix.png")
    plot_confusion_matrix(cm, CLASS_NAMES, cm_path)


if __name__ == "__main__":
    main()