"""Train the Task 1 convolutional VAE on preprocessed OASIS MRI slices."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

TASK1_DIR = Path(__file__).resolve().parent

from dataset import OASISDataset
from vae import LOGVAR_MAX, LOGVAR_MIN, VAE, vae_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train an OASIS convolutional VAE")
    parser.add_argument("--data-root", type=Path,
                        default=Path("/home/groups/comp3710/OASIS"))
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=TASK1_DIR / "results",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument(
        "--max-raw-logvar-magnitude", type=float, default=30.0,
        help="diagnostic warning threshold for unclamped raw logvar",
    )
    parser.add_argument("--max-mu-magnitude", type=float, default=1e6)
    parser.add_argument("--run-name", type=str, default="stable_v1")
    parser.add_argument("--skip-smoke-test", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(dataset, batch_size, shuffle, num_workers, seed):
    generator = torch.Generator().manual_seed(seed)
    options = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "generator": generator,
    }
    if num_workers > 0:
        options["persistent_workers"] = True
    return DataLoader(dataset, **options)


def _empty_stats():
    return {"min": float("inf"), "max": float("-inf"), "sum": 0.0, "count": 0}


def _update_stats(stats, values):
    values = values.detach()
    stats["min"] = min(stats["min"], values.min().item())
    stats["max"] = max(stats["max"], values.max().item())
    stats["sum"] += values.sum().item()
    stats["count"] += values.numel()


def _finish_stats(stats):
    if stats["count"] == 0:
        return None
    return {
        "min": stats["min"],
        "max": stats["max"],
        "mean": stats["sum"] / stats["count"],
    }


def _check_latents(mu, logvar, raw_logvar, max_mu_magnitude,
                   max_raw_logvar_magnitude):
    for name, values in (("mu", mu), ("logvar", logvar),
                         ("raw_logvar", raw_logvar)):
        if values is None or not torch.isfinite(values).all().item():
            raise FloatingPointError(f"{name} contains NaN or Inf")
    max_abs_mu = mu.detach().abs().max().item()
    max_abs_raw_logvar = raw_logvar.detach().abs().max().item()
    if max_abs_mu > max_mu_magnitude:
        raise FloatingPointError(
            f"mu magnitude {max_abs_mu:.3g} exceeded debug threshold "
            f"{max_mu_magnitude:.3g}"
        )
    # raw_logvar is monitored before clamping, so crossing this diagnostic
    # threshold is worth reporting but is not itself a numerical failure. The
    # effective logvar used by exp() remains bounded by LOGVAR_MIN/MAX.
    return max_abs_raw_logvar > max_raw_logvar_magnitude


def run_epoch(model, loader, device, beta, max_grad_norm,
              max_mu_magnitude, max_raw_logvar_magnitude, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = {"total": 0.0, "reconstruction": 0.0, "kl": 0.0}
    latent_stats = {
        "mu": _empty_stats(),
        "logvar": _empty_stats(),
        "raw_logvar": _empty_stats(),
    }
    gradient_stats = _empty_stats()
    raw_logvar_threshold_exceeded = False
    sample_count = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images in loader:
            images = images.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)

            reconstruction, mu, logvar = model(images)
            raw_logvar = model.encoder.last_raw_logvar
            if not torch.isfinite(reconstruction).all().item():
                raise FloatingPointError("reconstruction contains NaN or Inf")
            raw_logvar_threshold_exceeded |= _check_latents(
                mu, logvar, raw_logvar,
                max_mu_magnitude, max_raw_logvar_magnitude,
            )
            total_loss, reconstruction_loss, kl_loss = vae_loss(
                reconstruction, images, mu, logvar, beta,
            )
            if training:
                total_loss.backward()
                # PyTorch returns the total norm before clipping, which is the
                # useful value for monitoring whether gradients are growing.
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=max_grad_norm,
                )
                if not torch.isfinite(grad_norm).item():
                    raise FloatingPointError("gradient norm became NaN or Inf")
                _update_stats(gradient_stats, grad_norm.reshape(1))
                optimizer.step()

            _update_stats(latent_stats["mu"], mu)
            _update_stats(latent_stats["logvar"], logvar)
            _update_stats(latent_stats["raw_logvar"], raw_logvar)

            batch_size = images.size(0)
            totals["total"] += total_loss.detach().item() * batch_size
            totals["reconstruction"] += reconstruction_loss.detach().item() * batch_size
            totals["kl"] += kl_loss.detach().item() * batch_size
            sample_count += batch_size

    metrics = {name: value / sample_count for name, value in totals.items()}
    metrics["mu"] = _finish_stats(latent_stats["mu"])
    metrics["logvar"] = _finish_stats(latent_stats["logvar"])
    metrics["raw_logvar"] = _finish_stats(latent_stats["raw_logvar"])
    metrics["raw_logvar_threshold_exceeded"] = raw_logvar_threshold_exceeded
    metrics["gradient_norm"] = _finish_stats(gradient_stats)
    return metrics


def smoke_test(device, latent_dim, beta, learning_rate, max_grad_norm,
               max_mu_magnitude, max_raw_logvar_magnitude):
    """Verify one complete optimization step without changing the real model."""
    model = VAE(latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    images = torch.rand(2, 1, 256, 256, device=device)
    optimizer.zero_grad(set_to_none=True)
    reconstruction, mu, logvar = model(images)
    raw_logvar = model.encoder.last_raw_logvar
    raw_logvar_threshold_exceeded = _check_latents(
        mu, logvar, raw_logvar,
        max_mu_magnitude, max_raw_logvar_magnitude,
    )
    total_loss, reconstruction_loss, kl_loss = vae_loss(
        reconstruction, images, mu, logvar, beta,
    )
    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_grad_norm,
    )
    if not torch.isfinite(grad_norm).item():
        raise FloatingPointError("Smoke-test gradient norm became NaN or Inf")
    optimizer.step()
    assert reconstruction.shape == images.shape
    assert mu.shape == (images.size(0), latent_dim)
    assert logvar.shape == (images.size(0), latent_dim)
    assert torch.isfinite(reconstruction).all().item()
    assert torch.isfinite(mu).all().item()
    assert torch.isfinite(logvar).all().item()
    assert torch.isfinite(total_loss).item()
    print("\nSmoke test:")
    print(f"Recon loss:    {reconstruction_loss.item():.6f}")
    print(f"KL loss:       {kl_loss.item():.6f}")
    print(f"Total loss:    {total_loss.item():.6f}")
    print(f"mu range:      [{mu.min().item():.3f}, {mu.max().item():.3f}]")
    print(f"logvar range:  [{logvar.min().item():.3f}, {logvar.max().item():.3f}]")
    print(f"raw logvar range: [{raw_logvar.min().item():.3f}, "
          f"{raw_logvar.max().item():.3f}]")
    if raw_logvar_threshold_exceeded:
        print("Warning: raw logvar exceeded its diagnostic threshold; "
              "effective logvar is still clamped.")
    print(f"gradient norm (pre-clip): {grad_norm.item():.6f}")
    del model, optimizer, images
    if device.type == "cuda":
        torch.cuda.empty_cache()


def save_checkpoint(path, model, optimizer, epoch, args, validation_metrics):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": {
                "latent_dim": args.latent_dim,
                "beta": args.beta,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "max_grad_norm": args.max_grad_norm,
                "max_mu_magnitude": args.max_mu_magnitude,
                "raw_logvar_warning_threshold": args.max_raw_logvar_magnitude,
                "logvar_min": LOGVAR_MIN,
                "logvar_max": LOGVAR_MAX,
            },
            "validation_metrics": validation_metrics,
        },
        path,
    )


def main():
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.latent_dim <= 0:
        raise ValueError("epochs, batch_size, and latent_dim must be positive")
    if args.beta < 0.0:
        raise ValueError("beta must be non-negative")
    if args.learning_rate <= 0.0 or args.max_grad_norm <= 0.0:
        raise ValueError("learning_rate and max_grad_norm must be positive")
    if args.max_mu_magnitude <= 0.0 or args.max_raw_logvar_magnitude <= 0.0:
        raise ValueError("latent debug thresholds must be positive")
    if (not args.run_name or args.run_name in {".", ".."}
            or Path(args.run_name).name != args.run_name):
        raise ValueError("run_name must be one non-empty path component")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = args.results_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_dataset = OASISDataset(args.data_root / "keras_png_slices_train")
    validation_dataset = OASISDataset(args.data_root / "keras_png_slices_validate")
    print(f"Train images: {len(train_dataset)}")
    print(f"Validation images: {len(validation_dataset)}")

    train_loader = make_loader(
        train_dataset, args.batch_size, True, args.num_workers, args.seed,
    )
    validation_loader = make_loader(
        validation_dataset, args.batch_size, False, args.num_workers, args.seed,
    )

    if not args.skip_smoke_test:
        smoke_test(
            device, args.latent_dim, args.beta, args.learning_rate,
            args.max_grad_norm, args.max_mu_magnitude,
            args.max_raw_logvar_magnitude,
        )
        set_seed(args.seed)

    model = VAE(args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = []
    best_validation_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        train_metrics = run_epoch(
            model, train_loader, device, args.beta, args.max_grad_norm,
            args.max_mu_magnitude, args.max_raw_logvar_magnitude,
            optimizer=optimizer,
        )
        validation_metrics = run_epoch(
            model, validation_loader, device, args.beta, args.max_grad_norm,
            args.max_mu_magnitude, args.max_raw_logvar_magnitude,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - start_time

        record = {
            "epoch": epoch,
            "epoch_time": epoch_time,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)

        print(f"\nEpoch {epoch:02d}/{args.epochs}")
        print(f"Train total: {train_metrics['total']:.6f}")
        print(f"Train recon: {train_metrics['reconstruction']:.6f}")
        print(f"Train KL:    {train_metrics['kl']:.6f}")
        print(f"Val total:   {validation_metrics['total']:.6f}")
        print(f"Val recon:   {validation_metrics['reconstruction']:.6f}")
        print(f"Val KL:      {validation_metrics['kl']:.6f}")
        print(f"mu range:     [{train_metrics['mu']['min']:.3f}, "
              f"{train_metrics['mu']['max']:.3f}] "
              f"(mean {train_metrics['mu']['mean']:.3f})")
        print(f"logvar range: [{train_metrics['logvar']['min']:.3f}, "
              f"{train_metrics['logvar']['max']:.3f}] "
              f"(mean {train_metrics['logvar']['mean']:.3f})")
        print(f"raw logvar range: [{train_metrics['raw_logvar']['min']:.3f}, "
              f"{train_metrics['raw_logvar']['max']:.3f}] "
              f"(mean {train_metrics['raw_logvar']['mean']:.3f})")
        if (train_metrics["raw_logvar_threshold_exceeded"]
                or validation_metrics["raw_logvar_threshold_exceeded"]):
            largest_raw_logvar = max(
                abs(train_metrics["raw_logvar"]["min"]),
                abs(train_metrics["raw_logvar"]["max"]),
                abs(validation_metrics["raw_logvar"]["min"]),
                abs(validation_metrics["raw_logvar"]["max"]),
            )
            print(f"Warning: raw logvar magnitude reached "
                  f"{largest_raw_logvar:.3g}, above diagnostic threshold "
                  f"{args.max_raw_logvar_magnitude:.3g}; effective logvar "
                  f"remains clamped to [{LOGVAR_MIN:.0f}, {LOGVAR_MAX:.0f}].")
        print(f"Gradient norm (pre-clip): "
              f"mean {train_metrics['gradient_norm']['mean']:.3f}, "
              f"max {train_metrics['gradient_norm']['max']:.3f}")
        print(f"Epoch time:  {epoch_time:.2f}s")

        save_checkpoint(
            run_dir / "vae_last.pt",
            model, optimizer, epoch, args, validation_metrics,
        )
        if validation_metrics["total"] < best_validation_loss:
            best_validation_loss = validation_metrics["total"]
            save_checkpoint(
                run_dir / "vae_best.pt",
                model, optimizer, epoch, args, validation_metrics,
            )
            print("Saved new best validation checkpoint")

        with (run_dir / "vae_history.json").open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)

    print(f"\nBest validation total loss: {best_validation_loss:.6f}")
    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
