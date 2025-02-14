#!/usr/bin/env python3

"""
Example evaluation script for a fine-tuned Whisper-small model.
Computes WER on a test set and performs inference on a single audio file.

Usage (example):
    python eval.py \
        --model_dir ./whisper-small-finetuned \
        --test_csv /path/to/test.csv \
        --audio_file /path/to/single_audio.wav
"""

import argparse
import os

import torch
import librosa

from datasets import load_dataset
from evaluate import load as load_metric  # pip install evaluate
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration
)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and run inference using a fine-tuned Whisper-small model.")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the directory containing the fine-tuned model.")
    parser.add_argument("--test_csv", type=str, required=False,
                        help="Path to CSV with test data (audio_path, transcript).")
    parser.add_argument("--audio_file", type=str, required=False,
                        help="Path to a single audio file for quick transcription.")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Load fine-tuned model and processor
    processor = WhisperProcessor.from_pretrained(args.model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_dir)
    model.eval()  # set model to evaluation mode
    
    # If you have a GPU, move model to GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 2. (Optional) Evaluate on a test CSV to get WER
    if args.test_csv:
        wer_metric = load_metric("wer")
        
        test_data = load_dataset("csv", data_files=args.test_csv)["train"]
        
        def map_to_pred(examples):
            # Load and preprocess the audio
            audio, sr = librosa.load(examples["audio_path"], sr=16000)
            
            # Convert to log-mel spectrogram input
            input_features = processor.feature_extractor(
                audio, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(device)
            
            # Generate transcription
            with torch.no_grad():
                predicted_ids = model.generate(input_features)
            transcription = processor.tokenizer.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]
            
            return {"reference": examples["transcript"], "prediction": transcription}
        
        # Map test dataset to predictions
        results = test_data.map(map_to_pred)
        
        # Calculate WER
        wer_score = wer_metric.compute(
            references=results["reference"],
            predictions=results["prediction"]
        )
        print(f"Test WER: {wer_score:.3f}")

    # 3. (Optional) Transcribe a single audio file
    if args.audio_file:
        print(f"\nTranscribing file: {args.audio_file}")
        audio, sr = librosa.load(args.audio_file, sr=16000)
        
        input_features = processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)
        
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        transcription = processor.tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]
        
        print("Transcription:")
        print(transcription)

if __name__ == "__main__":
    main()
