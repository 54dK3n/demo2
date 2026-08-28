"""From-scratch convolutional VAE for 256x256 single-channel brain MRI."""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Bounding the encoder's log-variance prevents exp(logvar) from overflowing or
# producing an extreme sampling variance. Keep these values named so the same
# numerical policy is visible to training diagnostics and checkpoint metadata.
LOGVAR_MIN = -10.0
LOGVAR_MAX = 10.0


class Encoder(nn.Module):
    """Encode an MRI into the mean and log-variance of q(z|x)."""

    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.features = nn.Sequential(
            self._block(1, 32),       # [B, 1, 256, 256] -> [B, 32, 128, 128]
            self._block(32, 64),      # -> [B, 64, 64, 64]
            self._block(64, 128),     # -> [B, 128, 32, 32]
            self._block(128, 256),    # -> [B, 256, 16, 16]
        )
        flattened_dim = 256 * 16 * 16
        self.fc_mu = nn.Linear(flattened_dim, latent_dim)
        self.fc_logvar = nn.Linear(flattened_dim, latent_dim)
        self.last_raw_logvar = None

    @staticmethod
    def _block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, image):
        if image.ndim != 4 or image.shape[1:] != (1, 256, 256):
            raise ValueError(f"Expected [B, 1, 256, 256], got {list(image.shape)}")
        features = self.features(image).flatten(start_dim=1)
        mu = self.fc_mu(features)
        raw_logvar = self.fc_logvar(features)
        # Preserve a detached view for diagnostics while keeping the public
        # encode() interface unchanged.
        self.last_raw_logvar = raw_logvar.detach()
        logvar = torch.clamp(raw_logvar, min=LOGVAR_MIN, max=LOGVAR_MAX)
        assert mu.shape == (image.size(0), self.latent_dim)
        assert logvar.shape == (image.size(0), self.latent_dim)
        return mu, logvar


class Decoder(nn.Module):
    """Decode a latent vector into a normalized MRI reconstruction."""

    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Linear(latent_dim, 256 * 16 * 16)
        self.upsample = nn.Sequential(
            self._block(256, 128),    # [B, 256, 16, 16] -> [B, 128, 32, 32]
            self._block(128, 64),     # -> [B, 64, 64, 64]
            self._block(64, 32),      # -> [B, 32, 128, 128]
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid(),             # Match the input range [0, 1].
        )

    @staticmethod
    def _block(in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, latent):
        if latent.ndim != 2 or latent.size(1) != self.latent_dim:
            raise ValueError(f"Expected [B, {self.latent_dim}], got {list(latent.shape)}")
        features = self.fc(latent).view(latent.size(0), 256, 16, 16)
        reconstruction = self.upsample(features)
        assert reconstruction.shape == (latent.size(0), 1, 256, 256)
        return reconstruction


class VAE(nn.Module):
    """Combine the encoder, reparameterization trick, and decoder."""

    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    @staticmethod
    def reparameterize(mu, logvar):
        # Sampling epsilon separately keeps z differentiable with respect to mu/logvar.
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def encode(self, image):
        return self.encoder(image)

    def decode(self, latent):
        return self.decoder(latent)

    def forward(self, image):
        mu, logvar = self.encode(image)
        latent = self.reparameterize(mu, logvar)
        reconstruction = self.decode(latent)
        assert reconstruction.shape == image.shape
        return reconstruction, mu, logvar


def vae_loss(reconstruction, target, mu, logvar, beta=1.0):
    """Return batch-normalized total, reconstruction, and KL losses."""
    if reconstruction.shape != target.shape:
        raise ValueError("Reconstruction and target shapes must match")

    reconstruction_loss = F.mse_loss(
        reconstruction, target, reduction="mean",
    )

    # This coursework baseline normalizes both terms with a mean: reconstruction
    # over batch/pixels and KL over batch/latent dimensions. This makes logging
    # and beta tuning interpretable without mixing a pixel mean with a latent
    # sum; it is a baseline convention, not the only valid ELBO reduction.
    kl_elements = -0.5 * (
        1.0 + logvar - mu.pow(2) - logvar.exp()
    )
    kl_loss = kl_elements.mean()
    total_loss = reconstruction_loss + beta * kl_loss

    if not all(torch.isfinite(loss).item()
               for loss in (total_loss, reconstruction_loss, kl_loss)):
        raise FloatingPointError("VAE loss became non-finite")
    return total_loss, reconstruction_loss, kl_loss
