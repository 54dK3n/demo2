"""Run the live Part 3.2 demonstration: inference plus one training epoch."""

import argparse
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn as nn

import part3_2_resnet18 as experiment


DEFAULT_CHECKPOINT = Path("ResNet18_40epoch_target_best.pth")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the trained CIFAR-10 ResNet-18, then train a fresh "
            "ResNet-18 for one complete epoch and run inference again."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def make_model(device):
    model = experiment.ResNet18(num_classes=10).to(device)
    for module in model.modules():
        if isinstance(module, experiment.BasicBlock):
            nn.init.zeros_(module.bn2.weight)
    return model.to(memory_format=torch.channels_last)


def load_state_dict(path, device):
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Run part3_2_resnet18.py first, "
            "or pass --checkpoint with the path to a trained state dictionary."
        )
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    if not isinstance(payload, dict):
        raise TypeError("Expected a model state dictionary checkpoint")
    return payload


def make_optimizer(model, config):
    decay_parameters = [
        parameter for parameter in model.parameters() if parameter.ndim > 1
    ]
    no_decay_parameters = [
        parameter for parameter in model.parameters() if parameter.ndim <= 1
    ]
    parameter_groups = [
        {"params": decay_parameters, "weight_decay": config.weight_decay},
        {"params": no_decay_parameters, "weight_decay": 0.0},
    ]
    return torch.optim.SGD(
        parameter_groups,
        lr=config.learning_rate,
        momentum=experiment.MOMENTUM,
        nesterov=config.nesterov,
    )


def print_evaluation(label, result):
    loss, accuracy, correct, samples = result
    print(f"\n{label}")
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy * 100:.2f}% ({correct}/{samples})")


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This demonstration uses the GPU-resident CIFAR-10 pipeline and "
            "requires a CUDA GPU (run it on Rangpur)."
        )

    experiment.set_seed(experiment.SEED)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    config = replace(
        experiment.TARGET_CONFIGS[0],
        name="ResNet18_live_demo",
        group="Live demonstration",
        epochs=1,
        batch_size=args.batch_size,
    )
    train_loader, test_loader = experiment.create_cifar10_loaders(config, device)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    amp_enabled = experiment.USE_MIXED_PRECISION

    print("Device:", torch.cuda.get_device_name(0))
    print("Checkpoint:", args.checkpoint)

    trained_model = make_model(device)
    trained_model.load_state_dict(load_state_dict(args.checkpoint, device))
    print_evaluation(
        "1) Inference with the saved trained model",
        experiment.evaluate(
            trained_model,
            test_loader,
            criterion,
            device,
            amp_enabled,
            config.channels_last,
            config.use_tta,
        ),
    )

    del trained_model
    torch.cuda.empty_cache()

    experiment.set_seed(experiment.SEED)
    demo_model = make_model(device)
    optimizer = make_optimizer(demo_model, config)
    scheduler = experiment.build_scheduler(config, optimizer, len(train_loader))

    started = time.perf_counter()
    train_loss, train_accuracy, gpu_seconds = experiment.train_one_epoch(
        demo_model,
        train_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        amp_enabled,
        config.channels_last,
        config.augmentation,
        config.mixup_alpha,
    )
    elapsed = time.perf_counter() - started

    print("\n2) One complete training epoch with a fresh model")
    print(f"Training loss: {train_loss:.4f}")
    print(f"MixUp-weighted training accuracy: {train_accuracy * 100:.2f}%")
    print(f"Wall time: {elapsed:.2f}s (GPU kernels: {gpu_seconds:.2f}s)")

    print_evaluation(
        "3) Inference after the live training epoch",
        experiment.evaluate(
            demo_model,
            test_loader,
            criterion,
            device,
            amp_enabled,
            config.channels_last,
            config.use_tta,
        ),
    )


if __name__ == "__main__":
    main()
