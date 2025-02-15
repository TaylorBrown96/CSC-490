import importlib.metadata
# Install required packages if not already installed
required  = {'transformers', 'datasets', 'librosa', 'accelerate', 'evaluate'}
installed = {pkg.metadata['Name'] for pkg in importlib.metadata.distributions()}
missing   = required - installed

if missing:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])

import argparse
import os
import sys
import subprocess
import warnings

import torch
import librosa
from datasets import load_dataset
from my_collator import MySpeechCollator

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

# Suppress Hugging Face, Librosa, and PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Parse command-line arguments for training
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper-small on a custom dataset.")
    parser.add_argument(
        "--train_csv", 
        type=str, 
        default="train.csv",
        help="Path to training CSV file. (default: train.csv)"
        )
    parser.add_argument(
        "--eval_csv", 
        type=str, 
        default="test.csv",
        help="Path to evaluation CSV file. (default: eval.csv)"
        )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./whisper-small-finetuned",
        help="Directory to store the fine-tuned model. (default: ./whisper-small-finetuned)"
        )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=10, 
        help="Number of training epochs. (default: 5)"
        )
    parser.add_argument(
        "--train_batch_size", 
        type=int, 
        default=4, 
        help="Training batch size per device. (default: 4)"
        )
    parser.add_argument(
        "--eval_batch_size", 
        type=int, 
        default=4, 
        help="Eval batch size per device. (default: 4)"
        )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=1e-5, 
        help="Learning rate. (default: 1e-5)"
        )
    parser.add_argument(
        "--save_steps", 
        type=int, 
        default=100, 
        help="Save checkpoint every X update steps. (default: 100)"
        )
    parser.add_argument(
        "--eval_steps", 
        type=int, 
        default=100, 
        help="Run evaluation every X steps. (default: 100)"
        )
    parser.add_argument(
        "--logging_steps", 
        type=int, 
        default=50, 
        help="Log every X steps. (default: 50)"
        )
    parser.add_argument(
        "--gradient_accumulation_steps", 
        type=int, 
        default=1,
        help="Number of gradient accumulation steps. (default: 1)"
        )
    parser.add_argument(
        "--max_audio_length", 
        type=int, 
        default=30,
        help="Max audio length (in seconds) for truncation. (default: 30)"
        )

    return parser.parse_args()

class MySpeechCollator:
    """
    Custom collator to handle:
      - Padded "input_features" for Whisper
      - Padded "labels" from the tokenizer
    """
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        # Separate out the audio features and labels
        input_features_list = [f["input_features"] for f in features]
        label_list = [f["labels"] for f in features]

        # Pad the audio input features
        batch_inputs = self.processor.feature_extractor.pad(
            {"input_features": input_features_list}, 
            return_tensors="pt"
        )

        # Pad the labels
        batch_labels = self.processor.tokenizer.pad(
            {"input_ids": label_list}, 
            return_tensors="pt", 
            padding=True
        )

        # Return dict with properly padded "input_features" and "labels"
        return {
            "input_features": batch_inputs["input_features"],
            "labels": batch_labels["input_ids"],
        }

def main():
    args = parse_args()
    
    # 1. Load the pre-trained model and processor
    model_name = "openai/whisper-small"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # 2. Load datasets from CSV
    train_data = load_dataset("csv", data_files=args.train_csv)["train"]
    eval_data  = load_dataset("csv", data_files=args.eval_csv)["train"]
    
    # 3. Preprocessing function
    def preprocess_function(examples):
        audio, sr = librosa.load(examples["audio_path"], sr=16000)

        # Truncate if longer than max_audio_length
        max_len_samples = args.max_audio_length * 16000
        if len(audio) > max_len_samples:
            audio = audio[:max_len_samples]
        
        # Extract log-mel spectrogram
        inputs = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
        
        # Encode transcript
        labels = processor.tokenizer(examples["transcript"])

        return {
            "input_features": inputs["input_features"][0], 
            "labels": labels["input_ids"]
        }
    
    # 4. Map over dataset
    train_dataset = train_data.map(preprocess_function, remove_columns=train_data.column_names)
    eval_dataset  = eval_data.map(preprocess_function, remove_columns=eval_data.column_names)
    
    # 5. Use our custom collator
    data_collator = MySpeechCollator(processor=processor)
    
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
    )
    
    # 7. Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=processor.tokenizer
    )
    
    # 8. Train
    trainer.train()
    
    # 9. Save final model and processor
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
