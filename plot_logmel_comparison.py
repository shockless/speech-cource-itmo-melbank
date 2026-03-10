import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchaudio

from melbanks import LogMelFilterBanks


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    wav_path = project_root / "hw1" / "data" / "SpeechCommands" / "speech_commands_v0.02" / "yes" / "004ae714_nohash_0.wav"
    output_dir = project_root / "hw1" / "outputs" / "logmel_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    waveform, sample_rate = torchaudio.load(str(wav_path))
    if sample_rate != 16000:
        raise ValueError(f"Expected 16000 Hz audio, got {sample_rate}")

    reference = torch.log(
        torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=80,
            power=2.0,
        )(waveform)
        + 1e-6
    )
    custom = LogMelFilterBanks(n_fft=400, samplerate=sample_rate, hop_length=160, n_mels=80)(waveform)
    difference = (reference - custom).abs()

    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(reference.squeeze(0).numpy(), aspect="auto", origin="lower")
    plt.title("torchaudio log mel")
    plt.xlabel("Frame")
    plt.ylabel("Mel bin")

    plt.subplot(1, 3, 2)
    plt.imshow(custom.squeeze(0).detach().numpy(), aspect="auto", origin="lower")
    plt.title("Custom LogMelFilterBanks")
    plt.xlabel("Frame")
    plt.ylabel("Mel bin")

    plt.subplot(1, 3, 3)
    plt.imshow(difference.squeeze(0).detach().numpy(), aspect="auto", origin="lower")
    plt.title("Absolute difference")
    plt.xlabel("Frame")
    plt.ylabel("Mel bin")

    plt.tight_layout()
    plt.savefig(output_dir / "logmel_vs_torchaudio.png", dpi=200)
    plt.close()

    summary = {
        "wav_path": str(wav_path),
        "sample_rate": sample_rate,
        "shape_match": list(reference.shape) == list(custom.shape),
        "allclose": bool(torch.allclose(reference, custom)),
        "max_abs_diff": float(difference.max().item()),
        "mean_abs_diff": float(difference.mean().item()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
