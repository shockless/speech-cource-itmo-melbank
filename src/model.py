import torch
from torch import nn

from melbanks import LogMelFilterBanks


class SpeechCommandCNN(nn.Module):
    def __init__(self, n_mels: int = 80, groups: int = 1):
        super().__init__()
        self.log_mels = LogMelFilterBanks(n_mels=n_mels)
        self.encoder = nn.Sequential(
            nn.Conv1d(n_mels, 64, kernel_size=5, padding=2, groups=groups, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2, groups=groups, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 96, kernel_size=3, padding=1, groups=groups, bias=False),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 2),
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        features = self.log_mels(waveforms)
        encoded = self.encoder(features)
        return self.classifier(encoded)

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def flops(self, input_length: int = 16000, device: torch.device | None = None) -> int:
        if device is None:
            device = next(self.parameters()).device

        flops = 0
        hooks = []

        def conv1d_hook(module, inputs, output):
            nonlocal flops
            batch_size, out_channels, out_steps = output.shape
            kernel_mul_adds = module.kernel_size[0] * (module.in_channels // module.groups) * 2
            bias_ops = 1 if module.bias is not None else 0
            flops += batch_size * out_channels * out_steps * (kernel_mul_adds + bias_ops)

        def linear_hook(module, inputs, output):
            nonlocal flops
            batch_size = output.shape[0]
            bias_ops = 1 if module.bias is not None else 0
            flops += batch_size * module.out_features * (module.in_features * 2 + bias_ops)

        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                hooks.append(module.register_forward_hook(conv1d_hook))
            elif isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(linear_hook))

        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy_waveform = torch.zeros(1, input_length, device=device)
            _ = self(dummy_waveform)
        if was_training:
            self.train()

        for hook in hooks:
            hook.remove()
        return flops
