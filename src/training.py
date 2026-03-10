import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from src.data import create_dataloaders, use_cuda_transfer_optimizations
from src.model import SpeechCommandCNN


@dataclass
class TrainingConfig:
    data_root: str = "hw1/data"
    output_dir: str = "hw1/outputs/binary_speech_commands"
    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    n_mels: int = 80
    groups: int = 1
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this machine.")
        return torch.device("cuda")
    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available on this machine.")
        return torch.device("mps")
    if device_name == "cpu":
        return torch.device("cpu")
    return torch.device(device_name)


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for waveforms, labels in dataloader:
            waveforms = waveforms.to(device, non_blocking=use_cuda_transfer_optimizations(device))
            labels = labels.to(device, non_blocking=use_cuda_transfer_optimizations(device))

            logits = model(waveforms)
            loss = F.cross_entropy(logits, labels)

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += labels.size(0)

    return total_loss / total_examples, total_correct / total_examples


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0

    for waveforms, labels in dataloader:
        waveforms = waveforms.to(device, non_blocking=use_cuda_transfer_optimizations(device))
        labels = labels.to(device, non_blocking=use_cuda_transfer_optimizations(device))

        optimizer.zero_grad(set_to_none=True)
        logits = model(waveforms)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_examples += labels.size(0)

    return total_loss / total_examples


def run_training(config: TrainingConfig) -> tuple[dict, list[dict[str, float]]]:
    set_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(config.device)
    train_loader, valid_loader, test_loader = create_dataloaders(
        data_root=config.data_root,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        device=device,
    )

    model = SpeechCommandCNN(n_mels=config.n_mels, groups=config.groups).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    metrics_path = output_dir / "history.csv"
    checkpoint_path = output_dir / "best_model.pt"
    summary_path = output_dir / "summary.json"

    best_val_accuracy = 0.0
    history_rows = []

    print(f"Using device: {device}")

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        _, val_accuracy = evaluate(model, valid_loader, device)
        epoch_time = time.perf_counter() - epoch_start

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_accuracy": val_accuracy,
                "epoch_time_sec": epoch_time,
            }
        )

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} "
            f"val_accuracy={val_accuracy:.4f} "
            f"epoch_time_sec={epoch_time:.2f}"
        )

        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "best_val_accuracy": best_val_accuracy,
                },
                checkpoint_path,
            )

    with metrics_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["epoch", "train_loss", "val_accuracy", "epoch_time_sec"])
        writer.writeheader()
        writer.writerows(history_rows)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_accuracy = evaluate(model, test_loader, device)

    summary = {
        "device": str(device),
        "n_mels": config.n_mels,
        "groups": config.groups,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "num_parameters": model.num_parameters(),
        "flops": model.flops(device=device),
        "best_val_accuracy": best_val_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary, history_rows
