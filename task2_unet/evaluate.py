"""Evaluate the trained U-Net on the complete official OASIS test split."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap
from torch.utils.data import DataLoader

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
        description="Evaluate the U-Net on the official OASIS test split"
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-examples", type=int, default=5)
    return parser.parse_args()


def load_checkpoint(checkpoint_path, device):
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
        # Also accept a checkpoint that is directly a model state dictionary.
        model_state_dict = checkpoint
    else:
        raise KeyError(
            "Checkpoint does not contain 'model_state_dict' or 'model_state'"
        )

    return checkpoint, model_state_dict


def calculate_global_dice(intersections, prediction_counts, target_counts):
    dice_scores = []
    for class_id in range(NUM_CLASSES):
        denominator = (
            prediction_counts[class_id] + target_counts[class_id]
        ).item()
        if denominator == 0:
            dice_scores.append(None)
        else:
            dice = (
                2.0 * intersections[class_id].item() + EPSILON
            ) / (denominator + EPSILON)
            dice_scores.append(dice)
    return dice_scores


@torch.inference_mode()
def evaluate_test_set(model, loader, device):
    intersections = torch.zeros(
        NUM_CLASSES, dtype=torch.int64, device=device
    )
    prediction_counts = torch.zeros_like(intersections)
    target_counts = torch.zeros_like(intersections)

    for batch_index, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=device.type == "cuda")
        masks = masks.to(device, non_blocking=device.type == "cuda")

        logits = model(images)
        expected_logits_shape = (
            images.shape[0],
            NUM_CLASSES,
            256,
            256,
        )
        if tuple(logits.shape) != expected_logits_shape:
            raise ValueError(
                f"Expected logits shape {expected_logits_shape}, got "
                f"{tuple(logits.shape)} at test batch {batch_index}"
            )

        predictions = logits.argmax(dim=1)
        if predictions.shape != masks.shape:
            raise ValueError(
                f"Prediction shape {tuple(predictions.shape)} does not match "
                f"mask shape {tuple(masks.shape)}"
            )

        for class_id in range(NUM_CLASSES):
            predicted_class = predictions == class_id
            target_class = masks == class_id
            intersections[class_id] += (
                predicted_class & target_class
            ).sum()
            prediction_counts[class_id] += predicted_class.sum()
            target_counts[class_id] += target_class.sum()

    dice_scores = calculate_global_dice(
        intersections,
        prediction_counts,
        target_counts,
    )
    return (
        dice_scores,
        intersections.cpu(),
        prediction_counts.cpu(),
        target_counts.cpu(),
    )


def save_metrics_csv(
    output_path,
    dice_scores,
    intersections,
    prediction_counts,
    target_counts,
    checkpoint_path,
    number_of_samples,
):
    measured_scores = [score for score in dice_scores if score is not None]
    mean_dice = float(np.mean(measured_scores)) if measured_scores else None
    minimum_dice = min(measured_scores) if measured_scores else None

    checkpoint_record = checkpoint_path.resolve()
    try:
        checkpoint_record = checkpoint_record.relative_to(
            Path.cwd().resolve()
        )
    except ValueError:
        # Keep an external checkpoint absolute, but avoid embedding a personal
        # home path when the checkpoint is inside the repository.
        pass

    fieldnames = [
        "record",
        "class",
        "dice",
        "predicted_pixels",
        "ground_truth_pixels",
        "intersection",
        "value",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for class_id, dice in enumerate(dice_scores):
            writer.writerow(
                {
                    "record": "class",
                    "class": class_id,
                    "dice": "N/A" if dice is None else f"{dice:.10f}",
                    "predicted_pixels": int(prediction_counts[class_id]),
                    "ground_truth_pixels": int(target_counts[class_id]),
                    "intersection": int(intersections[class_id]),
                    "value": "",
                }
            )

        summary_rows = {
            "mean_dice": mean_dice,
            "minimum_dice": minimum_dice,
            "checkpoint": str(checkpoint_record),
            "number_of_test_samples": number_of_samples,
        }
        for name, value in summary_rows.items():
            writer.writerow(
                {
                    "record": "summary",
                    "class": name,
                    "dice": "",
                    "predicted_pixels": "",
                    "ground_truth_pixels": "",
                    "intersection": "",
                    "value": value,
                }
            )


def categorical_colormap():
    colors = ["#000000", "#377eb8", "#4daf4a", "#e41a1c"]
    colormap = ListedColormap(colors)
    normalizer = BoundaryNorm(
        boundaries=np.arange(-0.5, NUM_CLASSES + 0.5, 1),
        ncolors=NUM_CLASSES,
    )
    return colormap, normalizer


@torch.inference_mode()
def save_segmentation_examples(
    model,
    dataset,
    device,
    output_path,
    number_of_examples,
):
    number_of_examples = min(number_of_examples, len(dataset))
    indices = np.linspace(
        0,
        len(dataset) - 1,
        num=number_of_examples,
        dtype=int,
    )
    colormap, normalizer = categorical_colormap()
    figure, axes = plt.subplots(
        number_of_examples,
        4,
        figsize=(14, 3.2 * number_of_examples),
        squeeze=False,
    )

    for row, index in enumerate(indices):
        image, mask = dataset[int(index)]
        logits = model(image.unsqueeze(0).to(device))
        prediction = logits.argmax(dim=1)[0].cpu()
        error = prediction != mask

        axes[row, 0].imshow(image[0], cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, 0].set_title(f"Input MRI\n{dataset.image_paths[index].name}")
        axes[row, 1].imshow(mask, cmap=colormap, norm=normalizer)
        axes[row, 1].set_title("Ground truth")
        axes[row, 2].imshow(prediction, cmap=colormap, norm=normalizer)
        axes[row, 2].set_title("Prediction")
        axes[row, 3].imshow(error, cmap="gray", vmin=0, vmax=1)
        axes[row, 3].set_title("Error")

        for axis in axes[row]:
            axis.axis("off")

    figure.suptitle(
        "Official OASIS test-set segmentation examples",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def print_summary(
    checkpoint_path,
    number_of_samples,
    dice_scores,
):
    measured_scores = [score for score in dice_scores if score is not None]
    mean_dice = float(np.mean(measured_scores)) if measured_scores else None
    minimum_dice = min(measured_scores) if measured_scores else None

    print("\n" + "=" * 50)
    print("TEST SET EVALUATION")
    print("=" * 50)
    print("Checkpoint:", checkpoint_path)
    print("Number of test samples:", number_of_samples)
    print()

    for class_id, dice in enumerate(dice_scores):
        if dice is None:
            print(f"Class {class_id} DSC: N/A (absent from complete test set)")
        else:
            print(f"Class {class_id} DSC: {dice:.6f}")

    print()
    print(
        "Mean DSC:",
        "N/A" if mean_dice is None else f"{mean_dice:.6f}",
    )
    print(
        "Minimum DSC:",
        "N/A" if minimum_dice is None else f"{minimum_dice:.6f}",
    )
    print("\nAssignment threshold: DSC > 0.90 for every class\n")

    passes = []
    for class_id, dice in enumerate(dice_scores):
        passed = dice is not None and dice > 0.90
        passes.append(passed)
        print(f"Class {class_id} > 0.90: {'PASS' if passed else 'FAIL'}")

    print("\nOverall requirement:")
    print("PASS" if all(passes) else "FAIL")
    print("=" * 50)


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.num_examples <= 0:
        raise ValueError("--num-examples must be greater than zero")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    test_image_dir = args.data_root / "keras_png_slices_test"
    test_mask_dir = args.data_root / "keras_png_slices_seg_test"
    print("Test MRI directory:", test_image_dir)
    print("Test mask directory:", test_mask_dir)

    test_dataset = OASISSegmentationDataset(
        test_image_dir,
        test_mask_dir,
        validate_masks=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    checkpoint, model_state_dict = load_checkpoint(args.checkpoint, device)
    print("Checkpoint path:", args.checkpoint)
    if len(checkpoint) <= 20:
        print("Checkpoint keys:", sorted(checkpoint.keys()))
    else:
        print(
            "Checkpoint structure: direct model state dictionary with",
            len(checkpoint),
            "entries",
        )
    print("Checkpoint epoch:", checkpoint.get("epoch", "not available"))
    print(
        "Best validation minimum Dice:",
        checkpoint.get("best_min_dice", "not available"),
    )

    model = UNet(in_channels=1, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()

    (
        dice_scores,
        intersections,
        prediction_counts,
        target_counts,
    ) = evaluate_test_set(model, test_loader, device)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = RESULTS_DIR / "test_metrics.csv"
    examples_path = RESULTS_DIR / "test_segmentation_examples.png"

    save_metrics_csv(
        metrics_path,
        dice_scores,
        intersections,
        prediction_counts,
        target_counts,
        args.checkpoint,
        len(test_dataset),
    )
    save_segmentation_examples(
        model,
        test_dataset,
        device,
        examples_path,
        args.num_examples,
    )
    print_summary(args.checkpoint, len(test_dataset), dice_scores)
    print("Saved test metrics to:", metrics_path)
    print("Saved test examples to:", examples_path)


if __name__ == "__main__":
    main()
