"""Train a stable 64 x 64 single-channel DCGAN on OASIS MRI slices."""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset import OASISMRIDataset
from model import (
    LATENT_DIM,
    Discriminator,
    Generator,
    initialize_dcgan_weights,
    verify_model_shapes,
)


TASK_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE_DIR = Path("/home/groups/comp3710/OASIS/keras_png_slices_train")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the OASIS 64x64 DCGAN")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--results-dir", type=Path, default=TASK_DIR / "results")
    parser.add_argument("--run-name", default="dcgan_64_baseline")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Optional debugging limit per epoch")
    parser.add_argument("--skip-smoke-test", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, args, device):
    generator = torch.Generator().manual_seed(args.seed)
    options = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        generator=generator,
    )
    if args.num_workers > 0:
        options["persistent_workers"] = True
    return DataLoader(**options)


def set_requires_grad(model, enabled):
    for parameter in model.parameters():
        parameter.requires_grad_(enabled)


def ensure_finite(name, value):
    if not torch.isfinite(value).all().item():
        raise FloatingPointError(f"{name} contains NaN or Inf")


def train_batch(generator, discriminator, real, criterion, optimizer_g,
                optimizer_d, latent_dim):
    """Perform one discriminator update followed by one generator update."""
    batch_size = real.shape[0]
    device = real.device
    real_targets = torch.ones(batch_size, device=device)
    fake_targets = torch.zeros(batch_size, device=device)

    # Discriminator update: fake images are detached, so G is not updated here.
    set_requires_grad(discriminator, True)
    optimizer_d.zero_grad(set_to_none=True)
    real_logits = discriminator(real)
    noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
    fake = generator(noise)
    fake_logits_d = discriminator(fake.detach())
    loss_d_real = criterion(real_logits, real_targets)
    loss_d_fake = criterion(fake_logits_d, fake_targets)
    loss_d = loss_d_real + loss_d_fake
    ensure_finite("discriminator loss", loss_d)
    loss_d.backward()
    optimizer_d.step()

    # Generator update: freeze D parameters while retaining its input gradient.
    # D remains in training mode so its BatchNorm behavior stays consistent.
    set_requires_grad(discriminator, False)
    optimizer_g.zero_grad(set_to_none=True)
    fake_logits_g = discriminator(fake)
    loss_g = criterion(fake_logits_g, real_targets)
    ensure_finite("generator loss", loss_g)
    loss_g.backward()
    optimizer_g.step()
    set_requires_grad(discriminator, True)

    return {
        "loss_d": loss_d.detach().item(),
        "loss_d_real": loss_d_real.detach().item(),
        "loss_d_fake": loss_d_fake.detach().item(),
        "loss_g": loss_g.detach().item(),
        "d_real_pre_d": torch.sigmoid(real_logits.detach()).mean().item(),
        "d_fake_pre_d": torch.sigmoid(fake_logits_d.detach()).mean().item(),
        "d_fake_post_d": torch.sigmoid(fake_logits_g.detach()).mean().item(),
    }


def smoke_test(device, latent_dim):
    """Verify shapes, finite losses, and both alternating parameter updates."""
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)
    generator.apply(initialize_dcgan_weights)
    discriminator.apply(initialize_dcgan_weights)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()
    real = torch.rand(2, 1, 64, 64, device=device).mul_(2).sub_(1)
    g_before = generator.network[0].weight.detach().clone()
    d_before = discriminator.network[0].weight.detach().clone()
    metrics = train_batch(
        generator, discriminator, real, criterion, optimizer_g, optimizer_d, latent_dim
    )
    assert not torch.equal(g_before, generator.network[0].weight.detach())
    assert not torch.equal(d_before, discriminator.network[0].weight.detach())
    assert all(np.isfinite(value) for value in metrics.values())
    fake_shape, logits_shape = verify_model_shapes(latent_dim, 2, device)
    print(f"Smoke test passed: G {tuple(fake_shape)}, D {tuple(logits_shape)}")


@torch.no_grad()
def sample_and_measure(generator, fixed_noise, output_path):
    """Save fixed-noise evidence and return simple diversity diagnostics."""
    was_training = generator.training
    generator.eval()
    generated = generator(fixed_noise).cpu()
    save_image(generated, output_path, nrow=8, normalize=True, value_range=(-1, 1))

    flat = generated.flatten(1)
    if len(flat) > 1:
        distances = torch.cdist(flat, flat, p=2) / np.sqrt(flat.shape[1])
        distances.fill_diagonal_(float("inf"))
        nearest = distances.min(dim=1).values
        mean_pairwise = distances[torch.isfinite(distances)].mean().item()
        mean_nearest = nearest.mean().item()
        near_duplicate_rate = (nearest < 0.02).float().mean().item()
    else:
        mean_pairwise = mean_nearest = near_duplicate_rate = float("nan")

    generator.train(was_training)
    return {
        "sample_pixel_std": generated.std(dim=0, unbiased=False).mean().item(),
        "mean_pairwise_rmse": mean_pairwise,
        "mean_nearest_rmse": mean_nearest,
        "near_duplicate_rate": near_duplicate_rate,
    }


def save_loss_plot(history, path):
    epochs = [row["epoch"] for row in history]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, [row["loss_d"] for row in history], label="Discriminator")
    axis.plot(epochs, [row["loss_g"] for row in history], label="Generator")
    axis.set(xlabel="Epoch", ylabel="BCE loss", title="DCGAN training losses")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_discriminator_components_plot(history, path):
    epochs = [row["epoch"] for row in history]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, [row["loss_d_real"] for row in history], label="D real")
    axis.plot(epochs, [row["loss_d_fake"] for row in history], label="D fake")
    axis.set(
        xlabel="Epoch",
        ylabel="BCE loss",
        title="Discriminator loss components",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_history(history, path):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def checkpoint_payload(epoch, generator, discriminator, optimizer_g,
                       optimizer_d, args, fixed_noise, history):
    return {
        "epoch": epoch,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "generator_optimizer_state_dict": optimizer_g.state_dict(),
        "discriminator_optimizer_state_dict": optimizer_d.state_dict(),
        "fixed_noise": fixed_noise.cpu(),
        "history": history,
        "config": vars(args),
    }


def validate_args(args):
    positive = ("epochs", "batch_size", "latent_dim", "learning_rate",
                "sample_count", "checkpoint_every", "log_every")
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0 or args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("num-workers must be non-negative and max-batches positive")
    if not 0.0 <= args.beta1 < 1.0 or not 0.0 <= args.beta2 < 1.0:
        raise ValueError("Adam beta values must be in [0, 1)")
    if not args.run_name or Path(args.run_name).name != args.run_name:
        raise ValueError("run-name must be one non-empty path component")


def main():
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_dir = args.results_dir / args.run_name
    samples_dir = run_dir / "samples"
    checkpoints_dir = run_dir / "checkpoints"
    samples_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    if not args.skip_smoke_test:
        smoke_test(device, args.latent_dim)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    dataset = OASISMRIDataset(args.image_dir)
    loader = make_loader(dataset, args, device)
    print(f"Training MRI slices: {len(dataset)}")

    # Save unshuffled real-image evidence in the same [-1, 1] display mapping.
    real_examples = torch.stack([dataset[i] for i in range(min(64, len(dataset)))])
    save_image(real_examples, run_dir / "real_samples.png", nrow=8,
               normalize=True, value_range=(-1, 1))

    generator = Generator(args.latent_dim).to(device)
    discriminator = Discriminator().to(device)
    generator.apply(initialize_dcgan_weights)
    discriminator.apply(initialize_dcgan_weights)
    criterion = nn.BCEWithLogitsLoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.learning_rate,
                                   betas=(args.beta1, args.beta2))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate,
                                   betas=(args.beta1, args.beta2))
    fixed_noise = torch.randn(args.sample_count, args.latent_dim, 1, 1, device=device)

    initial_diversity = sample_and_measure(
        generator, fixed_noise, samples_dir / "epoch_000.png"
    )
    (run_dir / "initial_diagnostics.json").write_text(
        json.dumps(initial_diversity, indent=2) + "\n"
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        totals = {
            name: 0.0
            for name in (
                "loss_d",
                "loss_d_real",
                "loss_d_fake",
                "loss_g",
                "d_real_pre_d",
                "d_fake_pre_d",
                "d_fake_post_d",
            )
        }
        batches = 0
        generator.train()
        discriminator.train()
        for batch_index, real in enumerate(loader, start=1):
            real = real.to(device, non_blocking=device.type == "cuda")
            if real.shape[1:] != (1, 64, 64):
                raise ValueError(f"Bad real batch shape: {tuple(real.shape)}")
            metrics = train_batch(
                generator, discriminator, real, criterion,
                optimizer_g, optimizer_d, args.latent_dim,
            )
            for name, value in metrics.items():
                totals[name] += value
            batches += 1
            if batch_index % args.log_every == 0:
                print(
                    f"Epoch {epoch:03d}/{args.epochs:03d} batch {batch_index:04d} "
                    f"D_total={metrics['loss_d']:.4f} "
                    f"D_real={metrics['loss_d_real']:.4f} "
                    f"D_fake={metrics['loss_d_fake']:.4f} "
                    f"G={metrics['loss_g']:.4f}"
                )
            if args.max_batches is not None and batch_index >= args.max_batches:
                break

        row = {"epoch": epoch}
        row.update({name: value / batches for name, value in totals.items()})
        diversity = sample_and_measure(
            generator, fixed_noise, samples_dir / f"epoch_{epoch:03d}.png"
        )
        row.update(diversity)
        row["seconds"] = time.time() - started
        history.append(row)
        save_history(history, run_dir / "history.csv")
        save_loss_plot(history, run_dir / "loss_curve.png")
        save_discriminator_components_plot(
            history, run_dir / "discriminator_components.png"
        )

        payload = checkpoint_payload(
            epoch, generator, discriminator, optimizer_g, optimizer_d,
            args, fixed_noise, history,
        )
        torch.save(payload, checkpoints_dir / "latest.pt")
        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            torch.save(payload, checkpoints_dir / f"epoch_{epoch:03d}.pt")

        collapse_note = ""
        if row["near_duplicate_rate"] >= 0.5 or row["sample_pixel_std"] < 0.02:
            collapse_note = " WARNING: possible mode collapse; inspect samples."
        print(f"Epoch {epoch:03d}:")
        print(f"D_total={row['loss_d']:.4f}")
        print(f"D_real={row['loss_d_real']:.4f}")
        print(f"D_fake={row['loss_d_fake']:.4f}")
        print(f"G={row['loss_g']:.4f}")
        print(f"D(real,preD)={row['d_real_pre_d']:.3f}")
        print(f"D(fake,preD)={row['d_fake_pre_d']:.3f}")
        print(f"D(fake,postD)={row['d_fake_post_d']:.3f}")
        print(f"pairwise_RMSE={row['mean_pairwise_rmse']:.4f}")
        print(f"duplicates={row['near_duplicate_rate']:.1%}")
        print(f"time={row['seconds']:.1f}s{collapse_note}")

    print(f"Training evidence saved to: {run_dir}")


if __name__ == "__main__":
    main()
