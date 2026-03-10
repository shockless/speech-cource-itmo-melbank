from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_line(x_values, y_values, xlabel: str, ylabel: str, title: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_grid_lines(
    results: list[dict],
    x_key: str,
    series_key: str,
    y_key: str,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    series_values = sorted({result[series_key] for result in results})
    for series_value in series_values:
        subset = sorted(
            [result for result in results if result[series_key] == series_value],
            key=lambda item: item[x_key],
        )
        plt.plot(
            [item[x_key] for item in subset],
            [item[y_key] for item in subset],
            marker="o",
            label=f"{series_key}={series_value}",
        )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_heatmap(
    results: list[dict],
    n_mels_values: list[int],
    groups_values: list[int],
    metric_key: str,
    title: str,
    output_path: Path,
) -> None:
    grid = []
    for n_mels in n_mels_values:
        row = []
        for groups in groups_values:
            match = next(
                (
                    result[metric_key]
                    for result in results
                    if result["n_mels"] == n_mels and result["groups"] == groups
                ),
                None,
            )
            row.append(np.nan if match is None else match)
        grid.append(row)

    plt.figure(figsize=(9, 4.8))
    image = plt.imshow(grid, aspect="auto", cmap="viridis")
    plt.colorbar(image, shrink=0.9)
    plt.xticks(range(len(groups_values)), groups_values)
    plt.yticks(range(len(n_mels_values)), n_mels_values)
    plt.xlabel("groups")
    plt.ylabel("n_mels")
    plt.title(title)

    for row_index, row in enumerate(grid):
        for col_index, value in enumerate(row):
            if np.isnan(value):
                plt.text(col_index, row_index, "skip", ha="center", va="center", color="white")
            else:
                label = f"{value:.4f}" if value < 10 else f"{value:.2f}"
                plt.text(col_index, row_index, label, ha="center", va="center", color="white")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
