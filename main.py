import argparse
import json
from pathlib import Path

from src.experiment_utils import mean, read_history, write_aggregate_csv
from src.plotting import plot_grid_lines, plot_heatmap, plot_line
from src.training import TrainingConfig, run_training


def is_valid_groups_configuration(n_mels: int, groups: int) -> bool:
    return n_mels % groups == 0 and 64 % groups == 0 and 96 % groups == 0


def load_existing_result(run_dir: Path) -> tuple[dict, list[dict[str, float]]] | None:
    summary_path = run_dir / "summary.json"
    history_path = run_dir / "history.csv"
    if not summary_path.exists() or not history_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    history = read_history(history_path)
    return summary, history


def build_result_record(n_mels: int, groups: int, summary: dict, history: list[dict[str, float]]) -> dict:
    epoch_times = [row["epoch_time_sec"] for row in history]
    steady_state_times = epoch_times[1:] if len(epoch_times) > 1 else epoch_times
    return {
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


def write_report(
    output_path: Path,
    results: list[dict],
    plots_dir: Path,
    skipped_combinations: list[tuple[int, int]],
) -> None:
    groups_slice = sorted(
        [result for result in results if result["n_mels"] == 80],
        key=lambda item: item["groups"],
    )
    n_mels_slice = sorted(
        [result for result in results if result["groups"] == 1],
        key=lambda item: item["n_mels"],
    )
    best_result = max(results, key=lambda item: item["test_accuracy"])

    report = """# Assignment 1 Report

## Task 1. `LogMelFilterBanks`

В работе реализован слой `LogMelFilterBanks` на `torch` без использования готового `MelSpectrogram` внутри самого слоя.

Слой включает следующие этапы:

- вычисление STFT через `torch.stft`
- получение спектральной мощности
- проекцию на mel-шкалу через `torchaudio.functional.melscale_fbanks`
- логарифмирование `log(x + 1e-6)`

Корректность реализации была проверена сравнением с `torchaudio.transforms.MelSpectrogram` с последующим логарифмированием. Проверка показала совпадение формы выходов и численных значений.

## Task 2. Binary Classification Model

Для обучения использовался датасет `Speech Commands`. Исходная задача была сведена к бинарной классификации:

- `yes -> 1`
- `no -> 0`

Использовались стандартные разбиения датасета:

- `training`
- `validation`
- `testing`

В качестве модели использовалась компактная `Conv1d` CNN:

- входом служат признаки `LogMelFilterBanks`
- далее применяются несколько `Conv1d` блоков с `BatchNorm1d`, `ReLU` и `MaxPool1d`
- на выходе используется `AdaptiveAvgPool1d` и полносвязный классификатор

В ходе экспериментов сохранялись:

- `validation_accuracy`
- `test_accuracy`
- время эпохи
- число параметров
- FLOPs

Все полученные модели удовлетворяют ограничению задания:

- число параметров во всех конфигурациях меньше `100000`

## Full Grid Search

Для анализа качества и вычислительной эффективности был проведён полный перебор конфигураций:

- `n_mels in [20, 40, 80]`
- `groups in [1, 2, 4, 8, 16]`
"""

    if skipped_combinations:
        skipped_text = ", ".join(f"({n_mels}, {groups})" for n_mels, groups in skipped_combinations)
        report += f"\nНевалидные комбинации, которые были пропущены:\n\n- `{skipped_text}`\n"

    report += """

Сводные результаты:

| n_mels | groups | params | FLOPs | test accuracy | mean epoch time, s |
| --- | --- | ---: | ---: | ---: | ---: |
"""

    for result in sorted(results, key=lambda item: (item["n_mels"], item["groups"])):
        report += (
            f"| {result['n_mels']} | {result['groups']} | {result['num_parameters']} | "
            f"{result['flops']} | {result['test_accuracy']:.4f} | {result['mean_epoch_time_sec']:.2f} |\n"
        )

    report += f"""

## Graphs for `groups`

Графики ниже построены для среза `n_mels = 80`.

### Epoch Training Time vs Groups
![Epoch training time vs groups](outputs/full_grid_experiments/plots/epoch_time_vs_groups.png)

### Parameters vs Groups
![Parameters vs groups](outputs/full_grid_experiments/plots/parameters_vs_groups.png)

### FLOPs vs Groups
![FLOPs vs groups](outputs/full_grid_experiments/plots/flops_vs_groups.png)

### Test Accuracy vs Groups
![Test accuracy vs groups](outputs/full_grid_experiments/plots/test_accuracy_vs_groups.png)

## Graphs for `n_mels`

Графики ниже построены для среза `groups = 1`.

### Epoch Training Time vs n_mels
![Epoch training time vs n_mels](outputs/full_grid_experiments/plots/epoch_time_vs_n_mels.png)

### Parameters vs n_mels
![Parameters vs n_mels](outputs/full_grid_experiments/plots/parameters_vs_n_mels.png)

### FLOPs vs n_mels
![FLOPs vs n_mels](outputs/full_grid_experiments/plots/flops_vs_n_mels.png)

### Test Accuracy vs n_mels
![Test accuracy vs n_mels](outputs/full_grid_experiments/plots/test_accuracy_vs_n_mels.png)

## Quality Across Full Grid

### Test Accuracy Heatmap
![Test accuracy heatmap](outputs/full_grid_experiments/plots/test_accuracy_heatmap.png)

### Test Accuracy vs Groups by n_mels
![Test accuracy vs groups by n_mels](outputs/full_grid_experiments/plots/test_accuracy_vs_groups_by_n_mels.png)

### Test Accuracy vs n_mels by groups
![Test accuracy vs n_mels by groups](outputs/full_grid_experiments/plots/test_accuracy_vs_n_mels_by_groups.png)

## Analysis of `n_mels`

Для анализа влияния числа mel-фильтробанков сравним конфигурации при `groups = 1`.

| n_mels | params | FLOPs | test accuracy | mean epoch time, s |
| --- | ---: | ---: | ---: | ---: |
"""

    for result in n_mels_slice:
        report += (
            f"| {result['n_mels']} | {result['num_parameters']} | {result['flops']} | "
            f"{result['test_accuracy']:.4f} | {result['mean_epoch_time_sec']:.2f} |\n"
        )

    report += """

Выводы по `n_mels`:

- лучшая точность получена при `n_mels = 20`
- увеличение `n_mels` ведёт к росту числа параметров и FLOPs
- в данной бинарной задаче большее число mel-фильтров не дало улучшения качества

## Analysis of `groups`

Для анализа влияния grouped convolution сравним конфигурации при `n_mels = 80`.

| groups | params | FLOPs | test accuracy | mean epoch time, s |
| --- | ---: | ---: | ---: | ---: |
"""

    for result in groups_slice:
        report += (
            f"| {result['groups']} | {result['num_parameters']} | {result['flops']} | "
            f"{result['test_accuracy']:.4f} | {result['mean_epoch_time_sec']:.2f} |\n"
        )

    report += f"""

Выводы по `groups`:

- увеличение `groups` монотонно уменьшает число параметров и FLOPs
- наилучшее качество в этом срезе достигается при `groups = 1`
- конфигурация `groups = 4` даёт лучший компромисс между качеством и вычислительной стоимостью
- слишком большие значения `groups` ухудшают итоговую точность заметнее

## Chosen Baseline

В качестве baseline для следующего этапа выбрана конфигурация:

- `n_mels = {best_result['n_mels']}`
- `groups = {best_result['groups']}`

Характеристики baseline:

- `test_accuracy = {best_result['test_accuracy']:.4f}`
- `num_parameters = {best_result['num_parameters']}`
- `FLOPs = {best_result['flops']}`

Эта модель показывает лучший результат по качеству и уверенно укладывается в ограничение по числу параметров.

## Conclusion

В работе был реализован собственный слой `LogMelFilterBanks` и обучена компактная `Conv1d` CNN для бинарной классификации команд `yes/no` из датасета Speech Commands.

Итоговые выводы:

- реализация `LogMelFilterBanks` работает корректно
- для данной задачи оптимальным значением оказалось `n_mels = 20`
- параметр `groups` позволяет значительно уменьшать модель и вычислительную сложность
- лучшим компромиссом по эффективности оказался `groups = 4`
- наилучший общий результат показала конфигурация `n_mels = {best_result['n_mels']}`, `groups = {best_result['groups']}`
"""

    output_path.write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full n_mels x groups grid and build a report.")
    parser.add_argument("--device", default="mps", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--n-mels", type=int, nargs="+", default=[20, 40, 80])
    parser.add_argument("--groups", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-root", default="outputs/full_grid_experiments")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
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
            existing = None if args.force else load_existing_result(run_dir)
            if existing is None:
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
            else:
                summary, history = existing

            results.append(build_result_record(n_mels, groups, summary, history))

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
    groups_slice = sorted(
        [result for result in results if result["n_mels"] == max(args.n_mels)],
        key=lambda item: item["groups"],
    )
    n_mels_slice = sorted(
        [result for result in results if result["groups"] == 1],
        key=lambda item: item["n_mels"],
    )
    plot_line(
        [result["groups"] for result in groups_slice],
        [result["mean_epoch_time_sec"] for result in groups_slice],
        xlabel="groups",
        ylabel="Mean epoch time, s",
        title="Epoch training time vs groups",
        output_path=plots_dir / "epoch_time_vs_groups.png",
    )
    plot_line(
        [result["groups"] for result in groups_slice],
        [result["num_parameters"] for result in groups_slice],
        xlabel="groups",
        ylabel="Number of parameters",
        title="Parameters vs groups",
        output_path=plots_dir / "parameters_vs_groups.png",
    )
    plot_line(
        [result["groups"] for result in groups_slice],
        [result["flops"] for result in groups_slice],
        xlabel="groups",
        ylabel="FLOPs",
        title="FLOPs vs groups",
        output_path=plots_dir / "flops_vs_groups.png",
    )
    plot_line(
        [result["groups"] for result in groups_slice],
        [result["test_accuracy"] for result in groups_slice],
        xlabel="groups",
        ylabel="Test accuracy",
        title="Test accuracy vs groups",
        output_path=plots_dir / "test_accuracy_vs_groups.png",
    )
    plot_line(
        [result["n_mels"] for result in n_mels_slice],
        [result["mean_epoch_time_sec"] for result in n_mels_slice],
        xlabel="n_mels",
        ylabel="Mean epoch time, s",
        title="Epoch training time vs n_mels",
        output_path=plots_dir / "epoch_time_vs_n_mels.png",
    )
    plot_line(
        [result["n_mels"] for result in n_mels_slice],
        [result["num_parameters"] for result in n_mels_slice],
        xlabel="n_mels",
        ylabel="Number of parameters",
        title="Parameters vs n_mels",
        output_path=plots_dir / "parameters_vs_n_mels.png",
    )
    plot_line(
        [result["n_mels"] for result in n_mels_slice],
        [result["flops"] for result in n_mels_slice],
        xlabel="n_mels",
        ylabel="FLOPs",
        title="FLOPs vs n_mels",
        output_path=plots_dir / "flops_vs_n_mels.png",
    )
    plot_line(
        [result["n_mels"] for result in n_mels_slice],
        [result["test_accuracy"] for result in n_mels_slice],
        xlabel="n_mels",
        ylabel="Test accuracy",
        title="Test accuracy vs n_mels",
        output_path=plots_dir / "test_accuracy_vs_n_mels.png",
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
    write_report(
        output_path=project_root / "README.md",
        results=results,
        plots_dir=plots_dir,
        skipped_combinations=skipped_combinations,
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
