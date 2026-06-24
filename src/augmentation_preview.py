

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

from dataset import CIFAR10_MEAN, CIFAR10_STD, CLASS_NAMES

preview_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])


def unnormalize_safe(tensor):
    """Clamp tensor to [0,1] for safe display (used only when normalization is applied)."""
    return torch.clamp(tensor, 0, 1)


def main(num_examples=5, num_augmentations=5):
    raw_dataset = datasets.CIFAR10(root="./data", train=True, download=True)

    fig, axes = plt.subplots(num_examples, num_augmentations + 1, figsize=(15, 3 * num_examples))

    for row in range(num_examples):
        image, label = raw_dataset[row]  # PIL Image, int label

        # Column 0: original image
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"Original\n({CLASS_NAMES[label]})", fontsize=9)
        axes[row, 0].axis("off")

        # Remaining columns: augmented versions
        for col in range(1, num_augmentations + 1):
            augmented = preview_transform(image)
            axes[row, col].imshow(augmented)
            axes[row, col].set_title(f"Augmented #{col}", fontsize=9)
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig("outputs/augmentation_preview.png", dpi=150, bbox_inches="tight")
    print("Saved preview to outputs/augmentation_preview.png")
    plt.show()


if __name__ == "__main__":
    main()