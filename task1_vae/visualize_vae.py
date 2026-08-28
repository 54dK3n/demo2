"""Create reconstruction, loss, latent-manifold, and generation figures."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

TASK1_DIR = Path(__file__).resolve().parent
PART4_DIR = TASK1_DIR.parent
sys.path.insert(0, str(PART4_DIR))
sys.path.insert(0, str(TASK1_DIR))

from dataset import OASISDataset
from vae import VAE


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a trained OASIS VAE")
    parser.add_argument("--data-root", type=Path,
                        default=Path("/home/groups/comp3710/OASIS"))
    parser.add_argument("--results-dir", type=Path, default=PART4_DIR / "results")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-images", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def plot_loss_curves(history_path, output_path):
    with history_path.open("r", encoding="utf-8") as file:
        history = json.load(file)

    epochs = [record["epoch"] for record in history]
    names = (("total", "Total loss"),
             ("reconstruction", "Reconstruction loss"),
             ("kl", "KL loss"))
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, (key, title) in zip(axes, names):
        axis.plot(epochs, [record["train"][key] for record in history],
                  label="Train")
        axis.plot(epochs, [record["validation"][key] for record in history],
                  label="Validation")
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


@torch.no_grad()
def plot_reconstructions(model, loader, device, output_path, num_images):
    images = next(iter(loader))[:num_images].to(device)
    mu, _ = model.encode(images)
    reconstructions = model.decode(mu)
    assert reconstructions.shape == images.shape

    figure, axes = plt.subplots(2, num_images, figsize=(3 * num_images, 6))
    axes = np.asarray(axes).reshape(2, num_images)
    for index in range(num_images):
        axes[0, index].imshow(images[index, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        axes[0, index].set_title("Original")
        axes[1, index].imshow(reconstructions[index, 0].cpu(),
                              cmap="gray", vmin=0, vmax=1)
        axes[1, index].set_title("Reconstruction")
        axes[0, index].axis("off")
        axes[1, index].axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


@torch.no_grad()
def collect_latent_means(model, loader, device):
    latent_means = []
    for images in loader:
        mu, _ = model.encode(images.to(device, non_blocking=True))
        latent_means.append(mu.cpu())
    return torch.cat(latent_means).numpy()


def plot_umap_manifold(latent_means, output_path, seed):
    try:
        import umap
    except ImportError:
        print("UMAP skipped: install 'umap-learn' to create latent_manifold_umap.png")
        return False

    embedding = umap.UMAP(
        n_components=2, random_state=seed,
    ).fit_transform(latent_means)
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(embedding[:, 0], embedding[:, 1], s=12, alpha=0.7)
    axis.set_title("OASIS test-set latent manifold (UMAP of encoder mu)")
    axis.set_xlabel("UMAP dimension 1")
    axis.set_ylabel("UMAP dimension 2")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return True


@torch.no_grad()
def plot_random_generations(model, device, output_path, num_images, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    latent = torch.randn(
        num_images, model.latent_dim, generator=generator, device=device,
    )
    generated = model.decode(latent)
    assert generated.shape == (num_images, 1, 256, 256)

    figure, axes = plt.subplots(1, num_images, figsize=(3 * num_images, 3))
    axes = np.atleast_1d(axes)
    for index, axis in enumerate(axes):
        axis.imshow(generated[index, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        axis.set_title(f"Sample {index + 1}")
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main():
    args = parse_args()
    if args.num_images <= 0:
        raise ValueError("num_images must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint or args.results_dir / "vae_best.pt"
    checkpoint = load_checkpoint(checkpoint_path, device)
    latent_dim = int(checkpoint["config"]["latent_dim"])

    model = VAE(latent_dim).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    test_dataset = OASISDataset(args.data_root / "keras_png_slices_test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    if args.num_images > len(test_dataset):
        raise ValueError("num_images cannot exceed the test-set size")

    history_path = args.results_dir / "vae_history.json"
    plot_loss_curves(history_path, args.results_dir / "loss_curves.png")
    plot_reconstructions(
        model, test_loader, device,
        args.results_dir / "reconstructions.png", args.num_images,
    )
    latent_means = collect_latent_means(model, test_loader, device)
    plot_umap_manifold(
        latent_means, args.results_dir / "latent_manifold_umap.png", args.seed,
    )
    plot_random_generations(
        model, device, args.results_dir / "random_generations.png",
        args.num_images, args.seed,
    )
    print(f"Visualizations saved to: {args.results_dir}")


if __name__ == "__main__":
    main()
