"""64 x 64 single-channel DCGAN models for OASIS MRI generation."""

import torch
from torch import nn


LATENT_DIM = 128


class Generator(nn.Module):
    """Map ``[B, latent_dim, 1, 1]`` noise to grayscale MRI images."""

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.network = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise):
        if noise.ndim != 4 or noise.shape[1:] != (self.latent_dim, 1, 1):
            raise ValueError(
                f"Expected noise [B, {self.latent_dim}, 1, 1], got "
                f"{tuple(noise.shape)}"
            )
        generated = self.network(noise)
        assert generated.shape == (noise.shape[0], 1, 64, 64)
        return generated


class Discriminator(nn.Module):
    """Map a grayscale MRI batch to one raw real/fake logit per image."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
        )

    def forward(self, images):
        if images.ndim != 4 or images.shape[1:] != (1, 64, 64):
            raise ValueError(
                f"Expected images [B, 1, 64, 64], got {tuple(images.shape)}"
            )
        logits = self.network(images)
        assert logits.shape == (images.shape[0], 1, 1, 1)
        return logits.flatten()


def initialize_dcgan_weights(module):
    """Apply the initialization used by the original DCGAN baseline."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.zeros_(module.bias.data)


def verify_model_shapes(latent_dim=LATENT_DIM, batch_size=4, device="cpu"):
    """Run the requested Generator and Discriminator shape assertions."""
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)
    noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
    fake = generator(noise)
    logits = discriminator(fake)
    assert fake.shape == (batch_size, 1, 64, 64)
    assert logits.shape == (batch_size,)
    return fake.shape, logits.shape
