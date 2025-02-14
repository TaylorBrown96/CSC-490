import os
import csv
import argparse
import random

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate three CSV files (train, test, val) from 'audio' and 'transcripts' directories."
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="./audio",
        help="Path to the directory containing audio files (default: ./audio)."
    )
    parser.add_argument(
        "--transcript_dir",
        type=str,
        default="./transcripts",
        help="Path to the directory containing transcript files (default: ./transcripts)."
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default="train.csv",
        help="Name of the output CSV file for the training set (default: train.csv)."
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="test.csv",
        help="Name of the output CSV file for the test set (default: test.csv)."
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default="val.csv",
        help="Name of the output CSV file for the validation set (default: val.csv)."
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.7,
        help="Proportion of data to use for the training set (default: 0.7)."
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.2,
        help="Proportion of data to use for the test set (default: 0.2)."
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Proportion of data to use for the validation set (default: 0.1)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    audio_dir = args.audio_dir
    transcript_dir = args.transcript_dir
    train_csv = args.train_csv
    test_csv = args.test_csv
    val_csv = args.val_csv

    # Ensure splits add up to 1.0
    total_split = args.train_split + args.test_split + args.val_split
    if round(total_split, 2) != 1.0:
        raise ValueError(f"Splits must add up to 1.0 (Current: {total_split})")

    # Set the random seed for reproducibility
    random.seed(args.seed)

    if not os.path.isdir(audio_dir):
        raise ValueError(f"Audio directory does not exist: {audio_dir}")
    if not os.path.isdir(transcript_dir):
        raise ValueError(f"Transcript directory does not exist: {transcript_dir}")

    # Collect all matched audio/transcript pairs
    rows = []
    audio_files = sorted(os.listdir(audio_dir), key=lambda x: (
        int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else x
    ))

    for audio_file in audio_files:
        audio_path = os.path.join(audio_dir, audio_file)
        if not os.path.isfile(audio_path):
            continue

        base_name, _ = os.path.splitext(audio_file)
        transcript_file = base_name + ".txt"
        transcript_path = os.path.join(transcript_dir, transcript_file)

        if os.path.isfile(transcript_path):
            with open(transcript_path, "r", encoding="utf-8", errors="replace") as f:
                transcript_text = f.read().replace("�", "'").strip()
            rows.append({
                "audio_path": audio_path,
                "transcript": transcript_text
            })
        else:
            print(f"Warning: No matching transcript found for {audio_file}")

    # Shuffle the dataset before splitting
    random.shuffle(rows)

    # Compute split indices
    total_count = len(rows)
    train_count = int(total_count * args.train_split)
    test_count = int(total_count * args.test_split)
    val_count = total_count - (train_count + test_count)  # Remaining goes to val

    # Split the data
    train_rows = rows[:train_count]
    test_rows = rows[train_count:train_count + test_count]
    val_rows = rows[train_count + test_count:]

    # Define a helper function to write CSV files
    def write_csv(data, csv_path):
        fieldnames = ["audio_path", "transcript"]
        with open(csv_path, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow(row)

    # Write out the CSV files
    write_csv(train_rows, train_csv)
    write_csv(test_rows, test_csv)
    write_csv(val_rows, val_csv)

    print(f"Training CSV generated: {train_csv} (rows: {len(train_rows)})")
    print(f"Test CSV generated: {test_csv} (rows: {len(test_rows)})")
    print(f"Validation CSV generated: {val_csv} (rows: {len(val_rows)})")

if __name__ == "__main__":
    main()
