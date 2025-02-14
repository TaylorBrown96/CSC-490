#!/usr/bin/env python3

"""
Example training script for fine-tuning Whisper-small on a custom dataset.
Make sure you have the following installed:

    pip install torch torchvision torchaudio
    pip install transformers datasets librosa accelerate evaluate

Usage (example):
    python train.py \
        --train_csv /path/to/train.csv \
        --eval_csv /path/to/eval.csv \
        --output_dir ./whisper-small-finetuned
"""

import argparse
import os

import torch
import librosa

from datasets import Dataset, load_dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper-small on a custom dataset.")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to training CSV file.")
    parser.add_argument("--eval_csv", type=str, required=True, help="Path to evaluation CSV file.")
    parser.add_argument("--output_dir", type=str, default="./whisper-small-finetuned",
                        help="Directory to store the fine-tuned model.")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Training batch size per device.")
    parser.add_argument("--eval_batch_size", type=int, default=4, help="Eval batch size per device.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X update steps.")
    parser.add_argument("--eval_steps", type=int, default=100, help="Run evaluation every X steps.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X steps.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of gradient accumulation steps.")
    parser.add_argument("--max_audio_length", type=int, default=30,
                        help="Max audio length (in seconds) for truncation.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load the pre-trained model and processor
    model_name = "openai/whisper-small"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # 2. Create datasets from CSV
    # Expected columns in CSV: "audio_path" (or similar) and "transcript"
    # Example CSV format:
    #   audio_path,transcript
    #   /path/to/audio1.wav,"this is the first transcript"
    #   /path/to/audio2.wav,"this is the second transcript"
    #
    # Adapt column names to match your CSV structure if needed.
    train_data = load_dataset("csv", data_files=args.train_csv)["train"]
    eval_data  = load_dataset("csv", data_files=args.eval_csv)["train"]
    
    # 3. Define a preprocessing function
    def preprocess_function(examples):
        # Load and resample audio to 16kHz
        audio, sr = librosa.load(examples["audio_path"], sr=16000)

        # Truncate if longer than max_audio_length
        max_len_samples = args.max_audio_length * 16000
        if len(audio) > max_len_samples:
            audio = audio[:max_len_samples]
        
        # Extract log-mel spectrogram
        inputs = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
        
        # Tokenize transcript (labels)
        with processor.as_target_processor():
            labels = processor.tokenizer(examples["transcript"])
        
        return {
            "input_features": inputs["input_features"][0],  # shape [80, n_frames]
            "labels": labels["input_ids"]
        }
    
    # 4. Map the dataset through the preprocessing function
    train_dataset = train_data.map(preprocess_function, remove_columns=train_data.column_names)
    eval_dataset  = eval_data.map(preprocess_function, remove_columns=eval_data.column_names)
    
    # 5. Data collator to dynamically pad inputs and labels
    data_collator = DataCollatorForSeq2Seq(
        processor.tokenizer,
        model=model,
        return_tensors="pt",
        padding=True
    )
    
    # 6. Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy="steps",
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        fp16=torch.cuda.is_available(),  # use FP16 if GPU supports it
        predict_with_generate=True,
        generation_max_length=225,
        # You can add more parameters as needed
    )
    
    # 7. Create Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor  # or processor.tokenizer
    )
    
    # 8. Train!
    trainer.train()
    
    # 9. Save final model (and processor)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
