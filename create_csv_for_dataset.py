#!/usr/bin/env python3

"""
create_csv_for_dataset.py

Scan two directories named "audio" and "transcripts" and generate a CSV file
matching each audio file to its transcript. The script assumes:
  - Each audio file has a matching transcript file with the same base name.
  - "audio" directory contains files like: audio/<name>.wav
  - "transcripts" directory contains corresponding text files like: transcripts/<name>.txt
  - The transcript file is plain text.

Default behavior if no arguments are passed:
  --audio_dir      => './audio'
  --transcript_dir => './transcripts'
  --output_csv     => 'data.csv'

Usage Examples:
  (1) Rely on defaults:
      python create_csv_for_dataset.py

  (2) Override defaults:
      python create_csv_for_dataset.py \\
          --audio_dir /path/to/audio \\
          --transcript_dir /path/to/transcripts \\
          --output_csv my_data.csv
"""

import os
import csv
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a CSV file from 'audio' and 'transcripts' directories."
    )
    # Removed 'required=True' and added defaults.
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
        "--output_csv",
        type=str,
        default="data.csv",
        help="Name of the output CSV file (default: data.csv)."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    audio_dir = args.audio_dir
    transcript_dir = args.transcript_dir
    output_csv = args.output_csv

    # Ensure the directories exist
    if not os.path.isdir(audio_dir):
        raise ValueError(f"Audio directory does not exist: {audio_dir}")
    if not os.path.isdir(transcript_dir):
        raise ValueError(f"Transcript directory does not exist: {transcript_dir}")

    # Collect data
    rows = []

    # List all files in the audio directory
    audio_files = os.listdir(audio_dir)

    # For each audio file, look for a matching transcript
    for audio_file in audio_files:
        audio_path = os.path.join(audio_dir, audio_file)

        # Skip if not a normal file
        if not os.path.isfile(audio_path):
            continue

        # Derive base name (without extension)
        base_name, audio_ext = os.path.splitext(audio_file)

        # Construct transcript file name
        transcript_file = base_name + ".txt"
        transcript_path = os.path.join(transcript_dir, transcript_file)

        if os.path.isfile(transcript_path):
            # Read transcript
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript_text = f.read().strip()

            rows.append({
                "audio_path": audio_path,
                "transcript": transcript_text
            })
        else:
            print(f"Warning: No matching transcript found for {audio_file}")

    # Write CSV
    with open(output_csv, "w", encoding="utf-8", newline="") as csvfile:
        fieldnames = ["audio_path", "transcript"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"CSV generated: {output_csv}")
    print(f"Total matched pairs: {len(rows)}")

if __name__ == "__main__":
    main()
