#!/usr/bin/env python
"""
live_transcribe.py

Live transcription script that uses a fine-tuned Whisper model
but only logs system time (no model-based timestamps).

Dependencies:
  - sounddevice
  - numpy
  - transformers
  - torch

Usage:
  python live_transcribe.py
  Press Ctrl+C to exit.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import threading
import queue
import time
import re

import numpy as np
import sounddevice as sd
import torch
from transformers import pipeline

# Audio capture parameters
SAMPLE_RATE = 16000
CHUNK_DURATION = 5.0
CHANNELS = 1

audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio status:", status)
    audio_queue.put(indata.copy())

def record_audio():
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        callback=audio_callback
    ):
        print("Recording... Press Ctrl+C to stop.")
        while True:
            sd.sleep(1000)

def sanitize_english_text(text):
    """
    Optional function to remove characters outside typical English
    letters, digits, basic punctuation, and whitespace.
    """
    return re.sub(r"[^A-Za-z0-9\s,.!?'\"]", "", text)

def process_audio():
    """
    Collects audio chunks and transcribes them using your fine-tuned model.
    Instead of model timestamps, uses system time to log each chunk.
    """

    # Path to your locally fine-tuned Whisper model directory
    fine_tuned_path = "./whisper-small-en-finetuned"  # <--- ADJUST TO YOUR PATH

    device = 0 if torch.cuda.is_available() else -1

    # Build the pipeline WITHOUT model-based timestamps
    asr = pipeline(
        "automatic-speech-recognition",
        model=fine_tuned_path,  # Your fine-tuned model path
        device=device
    )

    frames_per_chunk = int(SAMPLE_RATE * CHUNK_DURATION)

    while True:
        frames = []
        frames_collected = 0

        # Collect enough audio to fill a chunk
        while frames_collected < frames_per_chunk:
            data = audio_queue.get()
            frames.append(data)
            frames_collected += data.shape[0]

        audio_chunk = np.concatenate(frames, axis=0).flatten()

        # Capture the current system time for this chunk
        chunk_time_str = time.strftime("%H:%M:%S", time.localtime())
        chunk_time_str_end = time.strftime("%H:%M:%S", time.localtime(time.time() + CHUNK_DURATION))

        # Transcribe using the pipeline
        try:
            result = asr(audio_chunk)
        except Exception as e:
            print(f"Error during transcription: {e}")
            continue

        # Print the transcription, labeling with system time
        raw_text = result.get("text", "")
        cleaned_text = sanitize_english_text(raw_text)
        print(f"\n[{chunk_time_str} - {chunk_time_str_end}] Transcription: {cleaned_text}")
        print("-" * 50)

        # Append to the output file with the same system time
        with open("LiveTranscription.txt", "a", encoding="utf-8") as f:
            f.write(f"[{chunk_time_str} - {chunk_time_str_end}] {cleaned_text}\n")

        # Optional small delay before next chunk
        time.sleep(0.1)

def main():
    with open("LiveTranscription.txt", "a", encoding="utf-8") as f:
        # Get the current date and store it as the header for that section as YYYY-MM-DD
        f.write(f"\nLive Transcription for {time.strftime('%Y-%m-%d', time.localtime())}\n")        
        f.write("-" * 50 + "\n")
        
    
    process_thread = threading.Thread(target=process_audio, daemon=True)
    process_thread.start()

    try:
        record_audio()
    except KeyboardInterrupt:
        print("\nStopping live transcription.")

if __name__ == "__main__":
    main()
