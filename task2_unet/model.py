"""A standard 2D U-Net for four-class brain MRI segmentation."""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Apply two 3x3 convolution, batch normalization, and ReLU blocks."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class DownBlock(nn.Module):
    """Downsample by two, then apply a DoubleConv block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return self.conv(x)


class UpBlock(nn.Module):
    """Upsample, concatenate the matching skip feature, then convolve."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        # After upsampling, decoder and skip tensors each have out_channels.
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor
    ) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat((x, skip), dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """Standard U-Net returning raw per-class segmentation logits."""

    def __init__(self, in_channels: int = 1, num_classes: int = 4) -> None:
        super().__init__()

        # Encoder
        self.encoder1 = DoubleConv(in_channels, 64)
        self.encoder2 = DownBlock(64, 128)
        self.encoder3 = DownBlock(128, 256)
        self.encoder4 = DownBlock(256, 512)

        # Bottleneck
        self.bottleneck_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
        self.decoder4 = UpBlock(1024, 512)
        self.decoder3 = UpBlock(512, 256)
        self.decoder2 = UpBlock(256, 128)
        self.decoder1 = UpBlock(128, 64)

        # One raw logit per class at every pixel.
        self.segmentation_head = nn.Conv2d(
            in_channels=64,
            out_channels=num_classes,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder features retained for skip connections.
        skip1 = self.encoder1(x)       # [B, 64, 256, 256]
        skip2 = self.encoder2(skip1)   # [B, 128, 128, 128]
        skip3 = self.encoder3(skip2)   # [B, 256, 64, 64]
        skip4 = self.encoder4(skip3)   # [B, 512, 32, 32]

        # Bottleneck.
        x = self.bottleneck_pool(skip4)
        x = self.bottleneck(x)         # [B, 1024, 16, 16]

        # Decoder: each block upsamples, concatenates its skip, and convolves.
        x = self.decoder4(x, skip4)    # [B, 512, 32, 32]
        x = self.decoder3(x, skip3)    # [B, 256, 64, 64]
        x = self.decoder2(x, skip2)    # [B, 128, 128, 128]
        x = self.decoder1(x, skip1)    # [B, 64, 256, 256]

        # No softmax or argmax: return raw logits for the loss function.
        logits = self.segmentation_head(x)
        return logits


if __name__ == "__main__":
    model = UNet(in_channels=1, num_classes=4)
    x = torch.randn(2, 1, 256, 256)

    model.eval()
    with torch.no_grad():
        output = model(x)

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Number of trainable parameters:", trainable_parameters)

    assert output.shape == (2, 4, 256, 256)
