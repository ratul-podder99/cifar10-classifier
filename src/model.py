

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two conv layers + BN + ReLU, followed by MaxPool (halves spatial size)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CIFAR10CNN(nn.Module):
    """
    Input  (3 x 32 x 32)
    -> ConvBlock(3   -> 64)   -> 16 x 16
    -> ConvBlock(64  -> 128)  -> 8 x 8
    -> ConvBlock(128 -> 256)  -> 4 x 4
    -> AdaptiveAvgPool2d(1)   -> 256 x 1 x 1
    -> Dropout -> Linear(256 -> num_classes)
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_model(num_classes: int = 10, dropout: float = 0.4) -> CIFAR10CNN:
    return CIFAR10CNN(num_classes=num_classes, dropout=dropout)


if __name__ == "__main__":
    # Sanity check: forward pass with a dummy batch + parameter count.
    model = build_model()
    dummy = torch.randn(4, 3, 32, 32)
    out = model(dummy)
    print(f"Output shape: {out.shape}")         
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")