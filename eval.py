import argparse
import os
import sys
import csv
import torch
import librosa
import warnings
import transformers

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration
)

# Install evaluate if missing
try:
    import evaluate
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "evaluate"])
    import evaluate

# Suppress Hugging Face, Librosa, and PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning)
transformers.logging.set_verbosity_error()

# Define command-line arguments for the script
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned Whisper-small model using a validation dataset.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./whisper-small-en-finetuned",
        help="Path to the directory containing the fine-tuned model. (default: ./whisper-small-finetuned)"
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default="val.csv",
        help="Path to the validation CSV file (default: val.csv)."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Number of samples to evaluate from val.csv (default: -1 for all)."
    )
    parser.add_argument(
        "--metrics_dir",
        type=str,
        default="./metrics/eval",
        help="Directory where metrics and predictions will be saved. (default: metrics)"
    )
    return parser.parse_args()

# Load validation samples from val.csv
def load_val_data(val_csv, num_samples=-1):
    if not os.path.exists(val_csv):
        print(f"\nError: Validation CSV file '{val_csv}' not found.\n")
        sys.exit(1)

    samples = []
    with open(val_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        samples = [row for row in reader]

    # If num_samples is -1, use all samples
    if num_samples == -1 or num_samples > len(samples):
        num_samples = len(samples)

    return samples[:num_samples]

# Transcribe an audio file using the model
def transcribe_audio(model, processor, audio_path, device):
    audio, sr = librosa.load(audio_path, sr=16000)

    # Convert to log-mel spectrogram input
    input_features = processor.feature_extractor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    transcription = processor.tokenizer.batch_decode(
        predicted_ids, skip_special_tokens=True
    )[0]

    return transcription

def main():
    # Parse command-line arguments
    args = parse_args()

    # Ensure model directory exists
    if not os.path.exists(args.model_dir):
        print(f"\nError: Model directory '{args.model_dir}' does not exist.\n")
        sys.exit(1)

    # Load model and processor
    print(f"Loading model from: {args.model_dir}")
    processor = WhisperProcessor.from_pretrained(args.model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_dir)
    model.eval()

    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Load validation data
    val_data = load_val_data(args.val_csv, args.num_samples)
    if not val_data:
        print("No samples found to evaluate. Exiting.")
        sys.exit(0)

    # Prepare output directories/files
    os.makedirs(args.metrics_dir, exist_ok=True)
    predictions_csv_path = os.path.join(args.metrics_dir, "predictions.csv")
    metrics_path = os.path.join(args.metrics_dir, "metrics.txt")

    # Prepare to compute WER
    wer_metric = evaluate.load("wer")

    # Lists to store references and predictions
    references = []
    predictions = []

    # Open CSV for writing predictions
    with open(predictions_csv_path, "w", encoding="utf-8", newline="") as csvfile:
        fieldnames = ["audio_path", "ground_truth", "prediction"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Evaluate each sample
        for sample in val_data:
            audio_path = sample["audio_path"]
            ground_truth = sample["transcript"]

            # Check if audio file exists
            if not os.path.exists(audio_path):
                print(f"\nWarning: Audio file '{audio_path}' not found. Skipping.\n")
                continue

            print(f"Evaluating audio file: {os.path.basename(audio_path)}")

            # Transcribe
            transcription = transcribe_audio(model, processor, audio_path, device)

            # Print to console
            print("Ground Truth Transcript:")
            print(ground_truth)
            print("\nModel Transcription:")
            print(transcription)
            print("-" * 50)

            # Log references & predictions for metric computation
            references.append(ground_truth)
            predictions.append(transcription)

            # Write to CSV
            writer.writerow({
                "audio_path": audio_path,
                "ground_truth": ground_truth,
                "prediction": transcription
            })

    # Compute WER on the entire evaluated set
    if references and predictions:
        wer_value = wer_metric.compute(predictions=predictions, references=references)
    else:
        wer_value = float("nan")

    # Write final metric to a text file
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Number of evaluated samples: {len(references)}\n")
        f.write(f"WER: {wer_value:.4f}\n")

    print(f"\nEvaluation complete. Results:\n- Predictions saved to: {predictions_csv_path}\n- Metrics saved to: {metrics_path}\n")
    print(f"WER for {len(references)} sample(s): {wer_value:.4f}")

if __name__ == "__main__":
    main()
