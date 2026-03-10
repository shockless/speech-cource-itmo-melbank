import csv
from pathlib import Path


def read_history(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return [
            {
                "epoch": int(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "val_accuracy": float(row["val_accuracy"]),
                "epoch_time_sec": float(row["epoch_time_sec"]),
            }
            for row in reader
        ]


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def write_aggregate_csv(results: list[dict], output_path: Path, fieldnames: list[str]) -> None:
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in fieldnames})
