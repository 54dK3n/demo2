"""Dataset for the Task 3 OASIS MRI DCGAN baseline."""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class OASISMRIDataset(Dataset):
    """Load verified grayscale MRI PNG slices as ``[1, 64, 64]`` tensors."""

    def __init__(self, image_dir):
        self.image_dir = Path(image_dir)
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"OASIS MRI directory not found: {self.image_dir}")

        self.image_paths = sorted(self.image_dir.glob("*.png"))
        if not self.image_paths:
            raise RuntimeError(f"No PNG images found in: {self.image_dir}")

        self.transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            # Do not silently convert masks, RGB images, or other unexpected data.
            if image.mode != "L":
                raise ValueError(
                    f"Expected PIL mode 'L', got {image.mode!r} for {image_path}"
                )
            if image.size != (256, 256):
                raise ValueError(
                    f"Expected image size (256, 256), got {image.size} for {image_path}"
                )
            tensor = self.transform(image)

        if tensor.shape != (1, 64, 64):
            raise ValueError(
                f"Expected tensor shape (1, 64, 64), got {tuple(tensor.shape)} "
                f"for {image_path}"
            )
        if tensor.min().item() < -1.0001 or tensor.max().item() > 1.0001:
            raise ValueError(f"Normalized values outside [-1, 1] for {image_path}")
        return tensor


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect the OASIS GAN dataset")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("/home/groups/comp3710/OASIS/keras_png_slices_train"),
    )
    args = parser.parse_args()
    dataset = OASISMRIDataset(args.image_dir)
    sample = dataset[0]
    print(f"Images: {len(dataset)}")
    print(f"First file: {dataset.image_paths[0].name}")
    print(f"Tensor: shape={tuple(sample.shape)}, dtype={sample.dtype}")
    print(f"Range: [{sample.min().item():.4f}, {sample.max().item():.4f}]")
