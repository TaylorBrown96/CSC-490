#!/usr/bin/env python

import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt

# Make sure 'evaluate' is installed. If not, install it on the fly.
try:
    import evaluate
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "evaluate"])
    import evaluate

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a WER plot from a CSV of ground_truth vs. predictions."
    )
    parser.add_argument(
        "--predictions_csv",
        type=str,
        default="metrics/predictions.csv",
        help="Path to the CSV file with columns: audio_path, ground_truth, prediction."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_export",
        help="Directory to save the WER plot. (default: eval_export)"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Check that the CSV exists
    if not os.path.isfile(args.predictions_csv):
        print(f"Error: Predictions CSV '{args.predictions_csv}' not found.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load the CSV
    rows = []
    with open(args.predictions_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        print(f"Error: No rows found in {args.predictions_csv}.")
        sys.exit(1)

    # Initialize Evaluate WER metric
    wer_metric = evaluate.load("wer")

    # We'll compute sample-level WER for each row
    sample_wers = []
    for i, row in enumerate(rows):
        ground_truth = row["ground_truth"]
        prediction   = row["prediction"]

        # Evaluate WER on a single sample
        # Evaluate expects list inputs
        try:
            sample_wer = wer_metric.compute(
                predictions=[prediction],
                references=[ground_truth]
            )
        except Exception as e:
            print(f"Warning: Failed to compute WER for row {i} due to: {e}")
            sample_wer = float("nan")

        sample_wers.append(sample_wer)

    # Compute the average WER
    numeric_wers = [w for w in sample_wers if not isinstance(w, str)]
    average_wer  = sum(numeric_wers) / len(numeric_wers) if numeric_wers else float("nan")

    # ---- PLOT ----
    plt.figure(figsize=(10, 6))
    plt.title("Word Error Rate (WER) per Sample")
    plt.xlabel("Sample Index")
    plt.ylabel("WER")

    x_vals = list(range(len(sample_wers)))

    # Plot bar chart of WERs
    plt.bar(x_vals, sample_wers, color="skyblue", label="WER per sample")

    # Add a horizontal line for the average WER
    if not (average_wer is None or average_wer != average_wer):  # NaN check
        plt.axhline(y=average_wer, color="red", linestyle="--", label=f"Average WER = {average_wer:.3f}")

    plt.legend()

    # Save the figure
    plot_path = os.path.join(args.output_dir, "wer_per_sample.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(f"\nPlot saved to: {plot_path}")
    print(f"Average WER over {len(sample_wers)} samples: {average_wer:.3f}")

if __name__ == "__main__":
    main()
