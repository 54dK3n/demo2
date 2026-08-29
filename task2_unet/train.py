"""Train the Task 2 U-Net on the official OASIS segmentation splits."""

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import UNet
from part4.dataset import OASISSegmentationDataset


# Straightforward baseline configuration.
NUM_CLASSES = 4
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
SEED = 42
EPSILON = 1e-6

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path("/home/groups/comp3710/OASIS")
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a four-class U-Net on OASIS MRI segmentation"
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    return parser.parse_args()


def set_seed(seed):
    """Reduce variation from initialization and DataLoader shuffling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def soft_dice_loss(logits, masks, num_classes=NUM_CLASSES, epsilon=EPSILON):
    """Compute differentiable mean Dice loss over all four classes."""
    # Cross entropy consumes raw logits, but Dice needs class probabilities.
    probabilities = torch.softmax(logits, dim=1)

    # one_hot starts as [B, H, W, C]; U-Net probabilities are [B, C, H, W].
    one_hot_masks = F.one_hot(masks, num_classes=num_classes)
    one_hot_masks = one_hot_masks.permute(0, 3, 1, 2).float()

    dimensions = (0, 2, 3)
    intersection = (probabilities * one_hot_masks).sum(dim=dimensions)
    probability_count = probabilities.sum(dim=dimensions)
    target_count = one_hot_masks.sum(dim=dimensions)

    dice_per_class = (
        2.0 * intersection + epsilon
    ) / (
        probability_count + target_count + epsilon
    )

    return 1.0 - dice_per_class.mean()


def combined_loss(logits, masks, criterion):
    """Combine categorical cross entropy and differentiable Dice loss."""
    # CrossEntropyLoss expects raw [B, C, H, W] logits and integer [B, H, W]
    # class targets. Softmax must not be applied before cross entropy.
    ce_loss = criterion(logits, masks)
    dice_loss = soft_dice_loss(logits, masks)
    return ce_loss + dice_loss


def validate_first_batch(logits, masks):
    expected_output_shape = (
        masks.shape[0],
        NUM_CLASSES,
        256,
        256,
    )
    if tuple(logits.shape) != expected_output_shape:
        raise ValueError(
            f"Expected model output shape {expected_output_shape}, got "
            f"{tuple(logits.shape)}"
        )

    minimum_target = masks.min().item()
    maximum_target = masks.max().item()
    if minimum_target < 0 or maximum_target >= NUM_CLASSES:
        raise ValueError(
            f"Target class IDs must be in [0, {NUM_CLASSES - 1}], got "
            f"minimum={minimum_target}, maximum={maximum_target}"
        )


def ensure_finite_gradients(model):
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"Non-finite gradient detected in {name}")


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    use_amp,
    check_gradients=False,
):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_index, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=use_amp)
        masks = masks.to(device, non_blocking=use_amp)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(images)
            if batch_index == 0:
                validate_first_batch(logits, masks)
            loss = combined_loss(logits, masks, criterion)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training loss at batch {batch_index}: "
                f"{loss.item()}"
            )

        scaler.scale(loss).backward()

        if check_gradients and batch_index == 0:
            # Unscale first so this checks the real gradients, not scaled FP16
            # gradients. The check is limited to the first training batch.
            scaler.unscale_(optimizer)
            ensure_finite_gradients(model)

        scaler.step(optimizer)
        scaler.update()

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    # Counts are accumulated over the complete validation split. This avoids
    # giving equal weight to batches with very different class distributions.
    intersections = torch.zeros(NUM_CLASSES, dtype=torch.float64, device=device)
    prediction_counts = torch.zeros_like(intersections)
    target_counts = torch.zeros_like(intersections)

    for batch_index, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=use_amp)
        masks = masks.to(device, non_blocking=use_amp)

        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(images)
            if batch_index == 0:
                validate_first_batch(logits, masks)
            loss = combined_loss(logits, masks, criterion)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite validation loss at batch {batch_index}: "
                f"{loss.item()}"
            )

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # Validation uses hard class predictions, unlike differentiable Dice
        # loss, because this measures the final categorical segmentation.
        predictions = logits.argmax(dim=1)
        for class_id in range(NUM_CLASSES):
            predicted_class = predictions == class_id
            target_class = masks == class_id
            intersections[class_id] += (
                predicted_class & target_class
            ).sum()
            prediction_counts[class_id] += predicted_class.sum()
            target_counts[class_id] += target_class.sum()

    dice_per_class = (
        2.0 * intersections + EPSILON
    ) / (
        prediction_counts + target_counts + EPSILON
    )

    per_class_dice = dice_per_class.cpu().tolist()
    mean_dice = float(dice_per_class.mean().item())
    minimum_dice = float(dice_per_class.min().item())
    validation_loss = total_loss / total_samples

    return validation_loss, per_class_dice, mean_dice, minimum_dice


def checkpoint_contents(
    epoch,
    model,
    optimizer,
    scheduler,
    best_min_dice,
    mean_dice,
    per_class_dice,
    configuration,
):
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_min_dice": best_min_dice,
        "mean_dice": mean_dice,
        "per_class_dice": per_class_dice,
        "configuration": configuration,
    }


def save_history(history, output_path):
    fieldnames = [
        "epoch",
        "train_loss",
        "validation_loss",
        "mean_validation_dice",
        "minimum_validation_dice",
        "class_0_dice",
        "class_1_dice",
        "class_2_dice",
        "class_3_dice",
        "learning_rate",
    ]
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_plots(history, results_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, [row["train_loss"] for row in history], label="Train")
    axis.plot(
        epochs,
        [row["validation_loss"] for row in history],
        label="Validation",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("CE + Dice loss")
    axis.set_title("U-Net training and validation loss")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(results_dir / "loss_curve.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    for class_id in range(NUM_CLASSES):
        axis.plot(
            epochs,
            [row[f"class_{class_id}_dice"] for row in history],
            label=f"Class {class_id}",
        )
    axis.axhline(0.90, color="black", linestyle="--", label="Target 0.90")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation Dice")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Per-class validation Dice")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(results_dir / "dice_curve.png", dpi=150)
    plt.close(figure)


def main():
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than zero")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")

    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print("Device:", device)
    if use_amp:
        print("GPU:", torch.cuda.get_device_name(0))
    print("AMP enabled:", use_amp)

    train_image_dir = args.data_root / "keras_png_slices_train"
    train_mask_dir = args.data_root / "keras_png_slices_seg_train"
    validation_image_dir = args.data_root / "keras_png_slices_validate"
    validation_mask_dir = args.data_root / "keras_png_slices_seg_validate"

    print("Training MRI directory:", train_image_dir)
    print("Training mask directory:", train_mask_dir)
    print("Validation MRI directory:", validation_image_dir)
    print("Validation mask directory:", validation_mask_dir)

    # Debug label validation belongs in dataset.py's smoke test. Normal
    # training avoids an expensive unique-value scan on every sample.
    train_dataset = OASISSegmentationDataset(
        train_image_dir,
        train_mask_dir,
        validate_masks=False,
    )
    validation_dataset = OASISSegmentationDataset(
        validation_image_dir,
        validation_mask_dir,
        validate_masks=False,
    )

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": use_amp,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_options,
    )

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(validation_dataset))

    model = UNet(in_channels=1, num_classes=NUM_CLASSES).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    configuration = {
        "num_classes": NUM_CLASSES,
        "batch_size": args.batch_size,
        "num_epochs": args.epochs,
        "learning_rate": args.lr,
        "weight_decay": WEIGHT_DECAY,
        "num_workers": args.num_workers,
        "seed": SEED,
        "amp": use_amp,
        "train_image_dir": str(train_image_dir),
        "train_mask_dir": str(train_mask_dir),
        "validation_image_dir": str(validation_image_dir),
        "validation_mask_dir": str(validation_mask_dir),
    }

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    history = []
    best_min_dice = -1.0
    final_mean_dice = 0.0
    final_per_class_dice = [0.0] * NUM_CLASSES

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            use_amp,
            check_gradients=epoch == 1,
        )
        (
            validation_loss,
            per_class_dice,
            mean_dice,
            minimum_dice,
        ) = validate(
            model,
            validation_loader,
            criterion,
            device,
            use_amp,
        )

        # Minimum class Dice directly reflects the requirement that every
        # label, rather than only the average label, must exceed 0.90.
        scheduler.step(minimum_dice)
        learning_rate = optimizer.param_groups[0]["lr"]
        epoch_seconds = time.perf_counter() - epoch_start

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "mean_validation_dice": mean_dice,
                "minimum_validation_dice": minimum_dice,
                "class_0_dice": per_class_dice[0],
                "class_1_dice": per_class_dice[1],
                "class_2_dice": per_class_dice[2],
                "class_3_dice": per_class_dice[3],
                "learning_rate": learning_rate,
            }
        )

        print(f"\nEpoch {epoch:02d}/{args.epochs}")
        print(f"Train Loss:      {train_loss:.6f}")
        print(f"Validation Loss: {validation_loss:.6f}")
        print("Validation Dice:")
        for class_id, dice in enumerate(per_class_dice):
            print(f"  Class {class_id}: {dice:.6f}")
        print(f"Mean Dice:       {mean_dice:.6f}")
        print(f"Minimum Dice:    {minimum_dice:.6f}")
        print(f"Learning Rate:   {learning_rate:.6g}")
        print(f"Epoch Time:      {epoch_seconds:.2f} seconds")

        if minimum_dice > best_min_dice:
            best_min_dice = minimum_dice
            torch.save(
                checkpoint_contents(
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    best_min_dice,
                    mean_dice,
                    per_class_dice,
                    configuration,
                ),
                CHECKPOINT_DIR / "best_unet.pt",
            )
            print("Saved new best checkpoint based on minimum class Dice.")

        if all(dice > 0.90 for dice in per_class_dice):
            print("Assignment DSC target reached: all classes > 0.90")

        final_mean_dice = mean_dice
        final_per_class_dice = per_class_dice

    torch.save(
        checkpoint_contents(
            args.epochs,
            model,
            optimizer,
            scheduler,
            best_min_dice,
            final_mean_dice,
            final_per_class_dice,
            configuration,
        ),
        CHECKPOINT_DIR / "last_unet.pt",
    )

    save_history(history, RESULTS_DIR / "training_history.csv")
    save_plots(history, RESULTS_DIR)

    print("\nTraining complete.")
    print("Best minimum validation Dice:", f"{best_min_dice:.6f}")
    print("Best checkpoint:", CHECKPOINT_DIR / "best_unet.pt")
    print("Last checkpoint:", CHECKPOINT_DIR / "last_unet.pt")
    print("Training history:", RESULTS_DIR / "training_history.csv")
    print("Loss plot:", RESULTS_DIR / "loss_curve.png")
    print("Dice plot:", RESULTS_DIR / "dice_curve.png")


if __name__ == "__main__":
    main()
