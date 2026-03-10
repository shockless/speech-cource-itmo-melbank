import argparse
import json
from pathlib import Path

from src.experiment_utils import mean, write_aggregate_csv
from src.plotting import plot_grid_lines, plot_heatmap
from src.training import TrainingConfig, run_training


def is_valid_groups_configuration(n_mels: int, groups: int) -> bool:
    return n_mels % groups == 0 and 64 % groups == 0 and 96 % groups == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full n_mels x groups grid and build a report.")
    parser.add_argument("--device", default="mps", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", default="hw1/data")
    parser.add_argument("--n-mels", type=int, nargs="+", default=[20, 40, 80])
    parser.add_argument("--groups", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-root", default="hw1/outputs/full_grid_experiments")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    output_root = project_root / args.output_root
    plots_dir = output_root / "plots"
    output_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    results = []
    skipped_combinations = []

    for n_mels in args.n_mels:
        for groups in args.groups:
            if not is_valid_groups_configuration(n_mels, groups):
                skipped_combinations.append((n_mels, groups))
                continue

            run_dir = output_root / f"n_mels_{n_mels}_groups_{groups}"
            summary, history = run_training(
                TrainingConfig(
                    data_root=args.data_root,
                    output_dir=str(run_dir),
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    n_mels=n_mels,
                    groups=groups,
                    num_workers=args.num_workers,
                    seed=args.seed,
                    device=args.device,
                )
            )

            epoch_times = [row["epoch_time_sec"] for row in history]
            steady_state_times = epoch_times[1:] if len(epoch_times) > 1 else epoch_times
            results.append(
                {
                    "n_mels": n_mels,
                    "groups": groups,
                    "history": history,
                    "num_parameters": summary["num_parameters"],
                    "flops": summary["flops"],
                    "best_val_accuracy": summary["best_val_accuracy"],
                    "test_accuracy": summary["test_accuracy"],
                    "test_loss": summary["test_loss"],
                    "mean_epoch_time_sec": mean(epoch_times),
                    "steady_state_epoch_time_sec": mean(steady_state_times),
                }
            )

    write_aggregate_csv(
        results,
        output_root / "aggregate_metrics.csv",
        [
            "n_mels",
            "groups",
            "num_parameters",
            "flops",
            "best_val_accuracy",
            "test_accuracy",
            "test_loss",
            "mean_epoch_time_sec",
            "steady_state_epoch_time_sec",
        ],
    )
    (output_root / "aggregate_metrics.json").write_text(json.dumps(results, indent=2))

    plot_heatmap(
        results=results,
        n_mels_values=args.n_mels,
        groups_values=args.groups,
        metric_key="test_accuracy",
        title="Test accuracy heatmap",
        output_path=plots_dir / "test_accuracy_heatmap.png",
    )
    plot_heatmap(
        results=results,
        n_mels_values=args.n_mels,
        groups_values=args.groups,
        metric_key="mean_epoch_time_sec",
        title="Mean epoch time heatmap",
        output_path=plots_dir / "epoch_time_heatmap.png",
    )
    plot_heatmap(
        results=results,
        n_mels_values=args.n_mels,
        groups_values=args.groups,
        metric_key="num_parameters",
        title="Parameters heatmap",
        output_path=plots_dir / "parameters_heatmap.png",
    )
    plot_heatmap(
        results=results,
        n_mels_values=args.n_mels,
        groups_values=args.groups,
        metric_key="flops",
        title="FLOPs heatmap",
        output_path=plots_dir / "flops_heatmap.png",
    )
    plot_grid_lines(
        results=results,
        x_key="groups",
        series_key="n_mels",
        y_key="test_accuracy",
        xlabel="groups",
        ylabel="Test accuracy",
        title="Test accuracy vs groups by n_mels",
        output_path=plots_dir / "test_accuracy_vs_groups_by_n_mels.png",
    )
    plot_grid_lines(
        results=results,
        x_key="n_mels",
        series_key="groups",
        y_key="test_accuracy",
        xlabel="n_mels",
        ylabel="Test accuracy",
        title="Test accuracy vs n_mels by groups",
        output_path=plots_dir / "test_accuracy_vs_n_mels_by_groups.png",
    )

    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "n_mels": args.n_mels,
                "groups": args.groups,
                "skipped_combinations": skipped_combinations,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
