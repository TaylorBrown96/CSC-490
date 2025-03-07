#!/usr/bin/env python
"""
adaptive_live_transcribe.py

Real-time audio transcription with an adaptive volume threshold
(similar to your GUI approach, but in a simple console script).

Dependencies:
  - sounddevice
  - numpy
  - transformers
  - torch
  - pydub

Usage:
  python adaptive_live_transcribe.py
  Press Ctrl+C to exit.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import threading
import queue
import time
import re
import os

import numpy as np
import sounddevice as sd
import torch
from transformers import pipeline

from pydub import AudioSegment

# -----------------------------
# 1) CONFIGURABLE PARAMS
# -----------------------------
SAMPLE_RATE = 16000
CHANNELS = 1

# Time-based chunking while capturing:
AUDIO_CHUNK_DURATION = 0.5  # seconds per small chunk
SILENCE_DURATION_THRESHOLD = 1.0  # seconds of silence to mark the end of an utterance
MAX_UTTERANCE_DURATION = 10.0  # max seconds to accumulate before forcing transcription

# Adaptive threshold parameters:
DEFAULT_THRESHOLD = -35.0   # lower bound for effective threshold (in dBFS)
AMBIENT_NOISE_LEVEL = -60.0 # initial guess for ambient noise (in dBFS)
ADAPTIVE_MARGIN = 5.0       # offset from ambient noise for detection
AMBIENT_ALPHA = 0.1         # smoothing factor for updating ambient noise

# Path to your locally fine-tuned Whisper model directory
FINE_TUNED_MODEL_PATH = "./whisper-small-en-finetuned"  # <--- Adjust if needed

# -----------------------------
# 2) QUEUES & LOG FILE
# -----------------------------
audio_queue = queue.Queue()
stop_event = threading.Event()

# Create a unique transcript filename for each run (YYYYMMDD_HHMMSS_Transcript.txt)
transcript_filename = time.strftime("%Y%m%d_%H%M%S_Transcript.txt")
if not os.path.exists(transcript_filename):
    with open(transcript_filename, "w", encoding="utf-8") as f:
        f.write(f"Live Transcription for {time.strftime('%Y-%m-%d', time.localtime())}\n")
        f.write("-" * 50 + "\n")

# -----------------------------
# 3) AUDIO CAPTURE CALLBACK
# -----------------------------
def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio status:", status)
    audio_queue.put(indata.copy())

def record_audio():
    """
    Continuously capture audio from the microphone and place
    frames into `audio_queue`.
    """
    with sd.InputStream(samplerate=SAMPLE_RATE,
                        channels=CHANNELS,
                        callback=audio_callback):
        print("Recording... Press Ctrl+C to stop.")
        while not stop_event.is_set():
            sd.sleep(100)

# -----------------------------
# 4) ASR PIPELINE
# -----------------------------
def load_asr_model():
    """
    Load the fine-tuned Whisper model as a pipeline.
    """
    device = 0 if torch.cuda.is_available() else -1
    asr = pipeline("automatic-speech-recognition",
                   model=FINE_TUNED_MODEL_PATH,
                   device=device)
    return asr

def sanitize_english_text(text):
    """
    Optional function to remove characters outside typical English
    letters, digits, basic punctuation, and whitespace.
    """
    return re.sub(r"[^A-Za-z0-9\s,.!?'\"]", "", text)

def float32_to_pydub(float_array, sample_rate=16000):
    """
    Convert float32 numpy array samples (-1.0..1.0) to
    a pydub AudioSegment (for dBFS measurements).
    """
    float_array = np.clip(float_array, -1.0, 1.0)
    int16_array = (float_array * 32767).astype(np.int16)
    segment = AudioSegment(
        data=int16_array.tobytes(),
        sample_width=2,  # 16 bits = 2 bytes
        frame_rate=sample_rate,
        channels=1
    )
    return segment

def process_audio(asr):
    """
    Continuously pulls small audio chunks from `audio_queue`, checks volume
    against an adaptive threshold to decide if we are in "speech" or "silence."
    Once speech ends, sends the collected audio to Whisper for transcription.
    """
    frames_per_chunk = int(SAMPLE_RATE * AUDIO_CHUNK_DURATION)
    max_frames = int(SAMPLE_RATE * MAX_UTTERANCE_DURATION)

    # State variables
    current_segment = []        # list of np arrays with speech frames
    silence_duration = 0.0
    ambient_noise_level = AMBIENT_NOISE_LEVEL

    while not stop_event.is_set():
        # 1. Collect frames for one chunk
        frames = []
        frames_collected = 0

        while frames_collected < frames_per_chunk and not stop_event.is_set():
            try:
                data = audio_queue.get(timeout=0.1)
                frames.append(data)
                frames_collected += data.shape[0]
            except queue.Empty:
                # No new audio in the queue just yet
                pass

        if stop_event.is_set():
            break

        audio_chunk = np.concatenate(frames, axis=0).flatten()
        segment_pd = float32_to_pydub(audio_chunk, SAMPLE_RATE)
        dbfs_val = segment_pd.dBFS

        # Adaptive threshold: do not drop below DEFAULT_THRESHOLD
        effective_threshold = max(DEFAULT_THRESHOLD, ambient_noise_level + ADAPTIVE_MARGIN)

        # Debug info
        print(f"Microphone dBFS: {dbfs_val:.2f} | Threshold: {effective_threshold:.2f}")

        # 2. Check if chunk is above threshold
        if dbfs_val > effective_threshold:
            # Speech
            current_segment.append(audio_chunk)
            silence_duration = 0.0
        else:
            # Silence
            # Update ambient noise estimate with exponential smoothing
            ambient_noise_level = (AMBIENT_ALPHA * dbfs_val) + ((1 - AMBIENT_ALPHA) * ambient_noise_level)
            silence_duration += AUDIO_CHUNK_DURATION

        # 3. If we have some speech frames, check if it's time to finalize
        if current_segment:
            total_frames = sum(len(chunk) for chunk in current_segment)

            # If enough silence or we exceed max utterance length
            if silence_duration >= SILENCE_DURATION_THRESHOLD or total_frames >= max_frames:
                # Build full utterance audio
                utterance_audio = np.concatenate(current_segment)
                duration_sec = len(utterance_audio) / SAMPLE_RATE

                chunk_time_str = time.strftime("%H:%M:%S", time.localtime())
                try:
                    result = asr(utterance_audio)
                    raw_text = result.get("text", "")
                    cleaned_text = sanitize_english_text(raw_text).strip()
                except Exception as e:
                    print(f"Error during transcription: {e}")
                    current_segment = []
                    silence_duration = 0.0
                    continue

                if cleaned_text:
                    # Log it
                    transcript_line = f"[{chunk_time_str} ~{duration_sec:.2f}s] {cleaned_text}"
                    print("Transcript:", transcript_line)
                    with open(transcript_filename, "a", encoding="utf-8") as f:
                        f.write(transcript_line + "\n")
                else:
                    print("Transcript: (empty after cleaning)")

                # Reset state
                current_segment = []
                silence_duration = 0.0


# -----------------------------
# 5) MAIN
# -----------------------------
def main():
    asr = load_asr_model()

    # Start the processing thread
    processing_thread = threading.Thread(target=process_audio, args=(asr,), daemon=True)
    processing_thread.start()

    try:
        record_audio()
    except KeyboardInterrupt:
        print("\nStopping live transcription...")
    finally:
        stop_event.set()
        processing_thread.join(timeout=2)
        print("Done.")

if __name__ == "__main__":
    main()
