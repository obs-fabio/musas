"""
Plot validation curves for all CV folds.

Example:
    python plot_training_curves.py \
        --input_dir ./results/domain/iemanja \
        --metric val_loss

Output:
    ./results/plots/iemanja.png
"""

import os
import glob
import argparse

import matplotlib.pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def smooth(values: list[float], smoothing: float = 0.9) -> list[float]:
    """Apply TensorBoard-like exponential smoothing."""

    if not values:
        return values

    smoothed = [values[0]]

    for value in values[1:]:
        smoothed_value = (
            smoothing * smoothed[-1]
            + (1.0 - smoothing) * value
        )
        smoothed.append(smoothed_value)

    return smoothed

def find_event_file(fold_dir: str) -> str | None:
    """Find the most recent TensorBoard event file inside a fold."""

    pattern = os.path.join(
        fold_dir,
        "**",
        "events.out.tfevents.*"
    )

    files = glob.glob(pattern, recursive=True)

    if not files:
        return None

    return max(files, key=os.path.getmtime)


def load_metric(event_file: str, metric: str):
    """Load a scalar metric from a TensorBoard event file."""

    accumulator = EventAccumulator(event_file)
    accumulator.Reload()

    available = accumulator.Tags()["scalars"]

    if metric not in available:
        raise ValueError(
            f"Metric '{metric}' not found.\n"
            f"Available metrics: {available}"
        )

    events = accumulator.Scalars(metric)

    values = [event.value for event in events]

    return values


def plot_folds(
    input_dir: str,
    output_dir: str,
    metric: str,
    smoothing: float
) -> None:

    fold_dirs = sorted(
        glob.glob(os.path.join(input_dir, "fold_*"))
    )

    if not fold_dirs:
        raise RuntimeError(
            f"No fold directories found in: {input_dir}"
        )

    plt.figure(figsize=(10, 6))

    plotted = 0

    for fold_dir in fold_dirs:

        fold_name = os.path.basename(fold_dir)

        event_file = find_event_file(fold_dir)

        if event_file is None:
            print(f"{fold_name}: no TensorBoard event file found.")
            continue

        try:
            values = load_metric(
                event_file,
                metric
            )
        except ValueError as error:
            print(f"{fold_name}: {error}")
            continue

        epochs = range(1, len(values) + 1)

        smoothed_values = smooth(
            values,
            smoothing=smoothing
        )

        plt.plot(
            epochs,
            smoothed_values,
            label=fold_name
        )

        plotted += 1

    if plotted == 0:
        raise RuntimeError(
            f"No curves were found for metric '{metric}'."
        )

    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.title(f"Validation curves — {metric}")

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend(
        ncol=2,
        fontsize=8
    )

    plt.tight_layout()

    # Create output directory if necessary
    os.makedirs(output_dir, exist_ok=True)

    # Use input directory name as output filename
    experiment_name = os.path.basename(
        os.path.normpath(input_dir)
    )

    output_file = os.path.join(
        output_dir,
        f"{experiment_name}.png"
    )

    plt.savefig(
        output_file,
        dpi=300
    )

    plt.close()

    print(f"Saved: {output_file}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing fold_00, fold_01, ..."
    )

    parser.add_argument(
        "--output_dir",
        default="./results/plots",
        help="Directory where the plot will be saved"
    )

    parser.add_argument(
        "--metric",
        default="val/loss_epoch",
        help="TensorBoard validation metric to plot"
    )

    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.9,
        help="Exponential smoothing factor (TensorBoard-like)"
    )

    args = parser.parse_args()

    plot_folds(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        metric=args.metric,
        smoothing=args.smoothing
    )


if __name__ == "__main__":
    main()