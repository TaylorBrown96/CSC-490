import argparse
import glob
import os

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read TensorBoard logs and create Matplotlib plots for scalar metrics."
    )
    parser.add_argument(
        "--logdir", 
        type=str, 
        required=True, 
        help="Path to the directory containing TensorBoard event files (events.out.tfevents.*)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./metrics_plots",
        help="Directory to save the output .png files. (default: ./metrics_plots)"
    )
    return parser.parse_args()


def sanitize_tag_to_filename(tag):
    """
    Replaces special characters in a TensorBoard tag
    so it can be safely used as part of a filename.
    """
    return tag.replace("/", "_").replace(" ", "_")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Find all event files recursively in logdir
    event_files = sorted(
        glob.glob(os.path.join(args.logdir, "**", "events.out.tfevents.*"), recursive=True)
    )
    if not event_files:
        print(f"No event files found in {args.logdir}")
        return

    # 2. Data structure to hold metrics: { tag: {event_file: [(step, value), ...]} }
    metrics_data = {}

    # 3. Parse each event file
    for ef in event_files:
        print(f"Parsing events from: {ef}")
        ea = event_accumulator.EventAccumulator(ef)
        ea.Reload()

        # All scalar tags in this file
        scalar_tags = ea.Tags().get("scalars", [])
        if not scalar_tags:
            print(f"No scalar tags found in {ef}\n")
            continue

        for tag in scalar_tags:
            scalar_events = ea.Scalars(tag)
            if tag not in metrics_data:
                metrics_data[tag] = {}
            # Store (step, value) pairs for this file
            metrics_data[tag][ef] = [(se.step, se.value) for se in scalar_events]

    # 4. Generate and save plots for each tag
    for tag, file_dict in metrics_data.items():
        plt.figure(figsize=(8, 6))
        plt.title(f"Metric: {tag}")
        plt.xlabel("Step")
        plt.ylabel("Value")

        # Each event file is a different line in the same plot
        for ef, step_values in file_dict.items():
            # Sort by step just in case
            step_values = sorted(step_values, key=lambda x: x[0])
            steps = [sv[0] for sv in step_values]
            values = [sv[1] for sv in step_values]

            # Shorten the file path in the legend
            label = os.path.basename(ef)  # or something more descriptive
            plt.plot(steps, values, label=label)

        plt.legend()
        
        # 5. Save figure as PNG
        tag_filename = sanitize_tag_to_filename(tag)
        out_path = os.path.join(args.output_dir, f"{tag_filename}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved plot for tag '{tag}' to {out_path}")


if __name__ == "__main__":
    main()
