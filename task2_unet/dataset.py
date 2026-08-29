"""Dataset utilities for the preprocessed OASIS MRI PNG slices."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class OASISSegmentationDataset(Dataset):
    """Load paired OASIS MRI slices and categorical segmentation masks.

    MRI filenames use ``case_<id>_slice_<n>.nii.png`` and their masks use
    ``seg_<id>_slice_<n>.nii.png``. If supplied, ``transform`` must accept
    ``(image, mask)`` tensors and return both tensors so that any future
    geometric operation is applied identically to the pair. Segmentation
    masks must always use nearest-neighbour interpolation.
    """

    RAW_TO_CLASS = {
        0: 0,
        85: 1,
        170: 2,
        255: 3,
    }

    def __init__(
        self,
        image_dir,
        mask_dir,
        transform=None,
        validate_masks=False,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.validate_masks = validate_masks

        if not self.image_dir.is_dir():
            raise FileNotFoundError(
                f"OASIS MRI image directory not found: {self.image_dir}"
            )
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(
                f"OASIS segmentation mask directory not found: {self.mask_dir}"
            )

        image_paths = sorted(self.image_dir.glob("*.png"))
        if not image_paths:
            raise RuntimeError(f"No MRI PNG files found in: {self.image_dir}")

        mask_paths_by_name = {}
        for mask_path in self.mask_dir.glob("*.png"):
            if mask_path.name in mask_paths_by_name:
                raise RuntimeError(
                    f"Duplicate mask filename found: {mask_path.name}"
                )
            mask_paths_by_name[mask_path.name] = mask_path

        if not mask_paths_by_name:
            raise RuntimeError(f"No mask PNG files found in: {self.mask_dir}")

        self.pairs = []
        used_mask_names = set()
        for image_path in image_paths:
            if not image_path.name.startswith("case_"):
                raise ValueError(
                    "Unexpected MRI filename. Expected a name beginning with "
                    f"'case_', but found: {image_path.name}"
                )

            mask_name = "seg_" + image_path.name[len("case_"):]
            mask_path = mask_paths_by_name.get(mask_name)
            if mask_path is None:
                raise FileNotFoundError(
                    f"Missing mask for MRI '{image_path.name}'. Expected: "
                    f"{self.mask_dir / mask_name}"
                )

            self.pairs.append((image_path, mask_path))
            used_mask_names.add(mask_name)

        extra_mask_names = sorted(
            set(mask_paths_by_name) - used_mask_names
        )
        if extra_mask_names:
            examples = ", ".join(extra_mask_names[:5])
            raise RuntimeError(
                f"Found {len(extra_mask_names)} mask file(s) without a "
                f"corresponding MRI image in {self.image_dir}. Examples: "
                f"{examples}"
            )

        # Exposed for straightforward inspection in the demonstration.
        self.image_paths = [image_path for image_path, _ in self.pairs]
        self.mask_paths = [mask_path for _, mask_path in self.pairs]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]

        with Image.open(image_path) as pil_image:
            if pil_image.mode != "L":
                raise ValueError(
                    f"Expected grayscale MRI mode 'L', got "
                    f"'{pil_image.mode}' for {image_path}"
                )
            image_array = np.array(pil_image, dtype=np.uint8, copy=True)

        if image_array.shape != (256, 256):
            raise ValueError(
                f"Expected MRI shape (256, 256), got {image_array.shape} "
                f"for {image_path}"
            )

        # The inspected MRI PNGs are uint8 with range 0--255.
        image = torch.from_numpy(image_array).to(torch.float32).div_(255.0)
        image = image.unsqueeze(0)

        with Image.open(mask_path) as pil_mask:
            if pil_mask.mode != "L":
                raise ValueError(
                    f"Expected grayscale mask mode 'L', got "
                    f"'{pil_mask.mode}' for {mask_path}"
                )
            raw_mask = np.array(pil_mask, dtype=np.uint8, copy=True)

        if raw_mask.shape != (256, 256):
            raise ValueError(
                f"Expected mask shape (256, 256), got {raw_mask.shape} "
                f"for {mask_path}"
            )

        if self.validate_masks:
            raw_values = set(np.unique(raw_mask).tolist())
            unexpected_values = sorted(
                raw_values - self.RAW_TO_CLASS.keys()
            )
            if unexpected_values:
                raise ValueError(
                    f"Unexpected mask values in {mask_path}: "
                    f"{unexpected_values}. Allowed values are "
                    f"{sorted(self.RAW_TO_CLASS)}."
                )

        # Explicit categorical conversion for CrossEntropyLoss targets.
        # Start with -1 so an unexpected raw value can never leave an
        # uninitialized class ID behind, even when debug validation is off.
        label_mask = np.full(raw_mask.shape, -1, dtype=np.int64)
        label_mask[raw_mask == 0] = 0
        label_mask[raw_mask == 85] = 1
        label_mask[raw_mask == 170] = 2
        label_mask[raw_mask == 255] = 3

        invalid_positions = label_mask == -1
        if invalid_positions.any():
            unexpected_values = np.unique(raw_mask[invalid_positions]).tolist()
            raise ValueError(
                f"Unexpected mask values in {mask_path}: "
                f"{unexpected_values}. Allowed values are "
                f"{sorted(self.RAW_TO_CLASS)}."
            )

        mask = torch.from_numpy(label_mask).long()

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        self._validate_output(image, mask, image_path, mask_path)
        return image, mask

    def _validate_output(self, image, mask, image_path, mask_path):
        if image.shape != (1, 256, 256):
            raise ValueError(
                f"Expected image tensor shape (1, 256, 256), got "
                f"{tuple(image.shape)} for {image_path}"
            )
        if mask.shape != (256, 256):
            raise ValueError(
                f"Expected mask tensor shape (256, 256), got "
                f"{tuple(mask.shape)} for {mask_path}"
            )
        if image.dtype != torch.float32:
            raise TypeError(
                f"Expected image dtype torch.float32, got {image.dtype} "
                f"for {image_path}"
            )
        if mask.dtype != torch.long:
            raise TypeError(
                f"Expected mask dtype torch.long, got {mask.dtype} "
                f"for {mask_path}"
            )

        if self.validate_masks:
            class_ids = set(torch.unique(mask).tolist())
            unexpected_class_ids = sorted(class_ids - {0, 1, 2, 3})
            if unexpected_class_ids:
                raise ValueError(
                    f"Unexpected class IDs in transformed mask {mask_path}: "
                    f"{unexpected_class_ids}. Allowed IDs are [0, 1, 2, 3]."
                )


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    data_root = Path("/home/groups/comp3710/OASIS")
    train_image_dir = data_root / "keras_png_slices_train"
    train_mask_dir = data_root / "keras_png_slices_seg_train"

    dataset = OASISSegmentationDataset(
        train_image_dir,
        train_mask_dir,
        validate_masks=True,
    )
    image, mask = dataset[0]

    print("Dataset size:", len(dataset))
    print("First MRI path:", dataset.image_paths[0])
    print("First mask path:", dataset.mask_paths[0])
    print("Image tensor shape:", image.shape)
    print("Image dtype:", image.dtype)
    print("Image min/max:", image.min().item(), image.max().item())
    print("Mask tensor shape:", mask.shape)
    print("Mask dtype:", mask.dtype)
    print("Mask unique class IDs:", torch.unique(mask).tolist())

    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch_images, batch_masks = next(iter(loader))
    print("Batch image shape:", batch_images.shape)
    print("Batch mask shape:", batch_masks.shape)

    assert batch_images.shape == (4, 1, 256, 256)
    assert batch_masks.shape == (4, 256, 256)

    # Check approximately 100 evenly spaced samples, including the final item.
    number_to_check = min(100, len(dataset))
    sample_indices = np.linspace(
        0,
        len(dataset) - 1,
        num=number_to_check,
        dtype=int,
    )
    allowed_class_ids = {0, 1, 2, 3}
    for index in sample_indices:
        _, sampled_mask = dataset[index]
        sampled_class_ids = set(torch.unique(sampled_mask).tolist())
        assert sampled_class_ids.issubset(allowed_class_ids), (
            f"Unexpected class IDs at dataset index {index}: "
            f"{sorted(sampled_class_ids - allowed_class_ids)}"
        )

    print(
        "Sampled mask validation:",
        f"passed for {number_to_check} samples",
    )
