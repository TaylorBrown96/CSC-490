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

# Suppress Hugging Face, Librosa, and PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning)
transformers.logging.set_verbosity_error()

# Define command-line arguments and help messages for the script
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned Whisper-small model using validation dataset.")
    parser.add_argument(
        "--model_dir", 
        type=str, 
        default="./whisper-small-finetuned",
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
        help="Number of samples to evaluate from val.csv (default: 5)."
        )
    return parser.parse_args()

# Load validation samples from val.csv
def load_val_data(val_csv, num_samples=5):
    if not os.path.exists(val_csv):
        print(f"\nError: Validation CSV file '{val_csv}' not found.\n")
        sys.exit(1)

    # Load validation samples from val.csv
    samples = []
    with open(val_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        samples = [row for row in reader]

    # Limit the number of samples to evaluate if specified
    if len(samples) < num_samples:
        print(f"\nWarning: Only found {len(samples)} samples in val.csv. Using all available samples.\n")
        num_samples = len(samples)

    return samples[:num_samples]

# Transcribe an audio file using the model
def transcribe_audio(model, processor, audio_path, device):
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Convert to log-mel spectrogram input
    input_features = processor.feature_extractor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device)
    
    # Generate transcription from input features
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    transcription = processor.tokenizer.batch_decode(
        predicted_ids, skip_special_tokens=True
    )[0]

    return transcription

def main():
    # Parse command-line arguments and load model
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

    # Load validation data from val.csv
    val_data = load_val_data(args.val_csv, args.num_samples)

    # Evaluate the model on each validation sample
    for sample in val_data:
        audio_path = sample["audio_path"]
        ground_truth = sample["transcript"]

        if not os.path.exists(audio_path):
            print(f"\nError: Audio file '{audio_path}' not found. Skipping.\n")
            continue

        print(f"\nEvaluating audio file: {os.path.basename(audio_path)}")

        # Transcribe the audio
        transcription = transcribe_audio(model, processor, audio_path, device)

        print("\nGround Truth Transcript:")
        print(ground_truth)

        print("\nModel Transcription:")
        print(transcription)
        print("-" * 50)

if __name__ == "__main__":
    main()
