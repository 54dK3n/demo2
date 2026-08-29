"""Dataset utilities for the preprocessed OASIS MRI PNG slices."""

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class OASISDataset(Dataset):
    """Load one normalized single-channel MRI image per item."""

    def __init__(self, image_dir):
        self.image_dir = Path(image_dir)
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"OASIS image directory not found: {self.image_dir}")

        # Sorting makes the train/validation/test sample order reproducible.
        self.image_paths = sorted(self.image_dir.glob("*.png"))
        if not self.image_paths:
            raise RuntimeError(f"No PNG images found in: {self.image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = read_image(str(self.image_paths[index]))
        if image.shape != (1, 256, 256):
            raise ValueError(
                f"Expected [1, 256, 256], got {list(image.shape)} for "
                f"{self.image_paths[index]}"
            )

        # The source PNG is uint8 in [0, 255]; the VAE consumes float32 in [0, 1].
        image = image.to(dtype=torch.float32).div_(255.0)
        assert 0.0 <= image.min().item() and image.max().item() <= 1.0
        return image
