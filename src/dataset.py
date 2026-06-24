
import torch
from torchvision import datasets, transforms

# CIFAR-10 per-channel mean/std 
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),       
        transforms.RandomHorizontalFlip(p=0.5),       
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    return train_transform, test_transform


def get_dataloaders(data_dir="./data", batch_size=128, num_workers=2):
    train_transform, test_transform = get_transforms()

    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders()
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")

    images, labels = next(iter(train_loader))
    print(f"Batch image shape: {images.shape}")  
    print(f"Batch label shape: {labels.shape}")    
    print(f"Sample labels: {labels[:10].tolist()}")