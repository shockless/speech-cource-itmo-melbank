from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchaudio.datasets import SPEECHCOMMANDS


TARGET_LABELS = {"no": 0, "yes": 1}


def pad_or_trim(waveform: torch.Tensor, target_num_samples: int) -> torch.Tensor:
    waveform = waveform.squeeze(0)
    if waveform.numel() > target_num_samples:
        return waveform[:target_num_samples]
    if waveform.numel() < target_num_samples:
        return F.pad(waveform, (0, target_num_samples - waveform.numel()))
    return waveform


class BinarySpeechCommands(Dataset):
    def __init__(
        self,
        root: str,
        subset: str,
        target_num_samples: int = 16000,
        download: bool = False,
    ):
        self.dataset = SPEECHCOMMANDS(root=root, subset=subset, download=download)
        self.target_num_samples = target_num_samples
        self.indices = [
            index
            for index in range(len(self.dataset))
            if self.dataset.get_metadata(index)[2] in TARGET_LABELS
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        waveform, sample_rate, label, speaker_id, utterance_number = self.dataset[self.indices[index]]
        if sample_rate != 16000:
            raise ValueError(f"Expected sample rate 16000, got {sample_rate}")
        waveform = pad_or_trim(waveform, self.target_num_samples)
        target = TARGET_LABELS[label]
        return waveform, target, speaker_id, utterance_number


def collate_batch(batch):
    waveforms, labels, _, _ = zip(*batch)
    return torch.stack(waveforms), torch.tensor(labels, dtype=torch.long)


def use_cuda_transfer_optimizations(device: torch.device) -> bool:
    return device.type == "cuda"


def create_dataloaders(
    data_root: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
):
    root_path = Path(data_root)
    root_path.mkdir(parents=True, exist_ok=True)

    train_dataset = BinarySpeechCommands(root=str(root_path), subset="training", download=True)
    valid_dataset = BinarySpeechCommands(root=str(root_path), subset="validation")
    test_dataset = BinarySpeechCommands(root=str(root_path), subset="testing")

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": use_cuda_transfer_optimizations(device),
        "persistent_workers": num_workers > 0,
        "collate_fn": collate_batch,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, valid_loader, test_loader
