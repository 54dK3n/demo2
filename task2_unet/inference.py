"""Run one deterministic OASIS test-slice segmentation for a live demo."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap

from dataset import OASISSegmentationDataset
from model import UNet


NUM_CLASSES = 4
EPSILON = 1e-6
TASK_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path("/home/groups/comp3710/OASIS")
DEFAULT_CHECKPOINT = TASK_DIR / "checkpoints" / "best_unet.pt"
RESULTS_DIR = TASK_DIR / "results"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Segment one deterministic MRI from the OASIS test split"
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    return parser.parse_args()


def load_model(checkpoint_path, device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Expected checkpoint dictionary, got {type(checkpoint).__name__}"
        )

    if "model_state_dict" in checkpoint:
        model_state_dict = checkpoint["model_state_dict"]
    elif "model_state" in checkpoint:
        model_state_dict = checkpoint["model_state"]
    elif checkpoint and all(
        isinstance(value, torch.Tensor) for value in checkpoint.values()
    ):
        model_state_dict = checkpoint
    else:
        raise KeyError(
            "Checkpoint does not contain 'model_state_dict' or 'model_state'"
        )

    model = UNet(in_channels=1, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()
    return model, checkpoint


def single_image_dice(prediction, target):
    dice_scores = []
    for class_id in range(NUM_CLASSES):
        predicted_class = prediction == class_id
        target_class = target == class_id
        predicted_pixels = predicted_class.sum().item()
        target_pixels = target_class.sum().item()
        denominator = predicted_pixels + target_pixels

        if denominator == 0:
            dice_scores.append(None)
        else:
            intersection = (predicted_class & target_class).sum().item()
            dice = (2.0 * intersection + EPSILON) / (
                denominator + EPSILON
            )
            dice_scores.append(dice)
    return dice_scores


def save_demo_figure(image, mask, prediction, output_path):
    colors = ["#000000", "#377eb8", "#4daf4a", "#e41a1c"]
    colormap = ListedColormap(colors)
    normalizer = BoundaryNorm(
        boundaries=np.arange(-0.5, NUM_CLASSES + 0.5, 1),
        ncolors=NUM_CLASSES,
    )
    error = prediction != mask

    figure, axes = plt.subplots(1, 4, figsize=(15, 4))
    axes[0].imshow(image[0], cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Input MRI")
    axes[1].imshow(mask, cmap=colormap, norm=normalizer)
    axes[1].set_title("Ground truth")
    axes[2].imshow(prediction, cmap=colormap, norm=normalizer)
    axes[2].set_title("Prediction")
    axes[3].imshow(error, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Error")

    for axis in axes:
        axis.axis("off")

    figure.suptitle("U-Net segmentation on an official OASIS test MRI")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    test_dataset = OASISSegmentationDataset(
        args.data_root / "keras_png_slices_test",
        args.data_root / "keras_png_slices_seg_test",
        validate_masks=False,
    )
    if args.index < 0 or args.index >= len(test_dataset):
        raise IndexError(
            f"--index must be between 0 and {len(test_dataset) - 1}, "
            f"got {args.index}"
        )

    print("\nLoading checkpoint:")
    print(args.checkpoint)
    model, checkpoint = load_model(args.checkpoint, device)
    if len(checkpoint) <= 20:
        print("Checkpoint keys:", sorted(checkpoint.keys()))
    if "epoch" in checkpoint:
        print("Checkpoint epoch:", checkpoint["epoch"])

    image, mask = test_dataset[args.index]
    image_path = test_dataset.image_paths[args.index]
    mask_path = test_dataset.mask_paths[args.index]
    batched_image = image.unsqueeze(0).to(device)

    print("\nTest MRI:")
    print(image_path.name)
    print("\nGround truth:")
    print(mask_path.name)
    print("\nInput shape:")
    print(batched_image.shape)

    with torch.inference_mode():
        logits = model(batched_image)
        prediction = logits.argmax(dim=1)

    expected_logits_shape = (1, NUM_CLASSES, 256, 256)
    if tuple(logits.shape) != expected_logits_shape:
        raise ValueError(
            f"Expected logits shape {expected_logits_shape}, got "
            f"{tuple(logits.shape)}"
        )

    print("\nOutput logits shape:")
    print(logits.shape)
    print("\nPrediction shape:")
    print(prediction.shape)

    prediction = prediction[0].cpu()
    print("\nPredicted classes:")
    print(torch.unique(prediction).tolist())
    print("\nGround-truth classes:")
    print(torch.unique(mask).tolist())

    dice_scores = single_image_dice(prediction, mask)
    print()
    for class_id, dice in enumerate(dice_scores):
        if dice is None:
            print(
                f"Class {class_id} DSC: "
                "N/A (class absent in prediction and ground truth)"
            )
        else:
            print(f"Class {class_id} DSC: {dice:.6f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "demo_prediction.png"
    save_demo_figure(image, mask, prediction, output_path)
    print("\nSaved demo result to:")
    print(output_path)


if __name__ == "__main__":
    main()
