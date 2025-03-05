import cv2
import tkinter as tk
from tkinter import Text, Label, Button, Frame
from PIL import Image, ImageTk

import threading
import queue
import time
import re
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import sounddevice as sd
import torch
from transformers import pipeline

# pydub for volume-based gating
from pydub import AudioSegment

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("CSC-490 Live Demo")

        # -----------------------------
        # 1) SETUP MAIN GUI COMPONENTS
        # -----------------------------
        self.main_frame = Frame(root, bg="#b0bec5", padx=10, pady=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Webcam feed
        self.video_label = Label(self.main_frame, bg="white", width=800, height=600)
        self.video_label.grid(row=0, column=0, padx=10, pady=10, rowspan=3)

        # Right panel
        self.right_panel = Frame(self.main_frame, bg="#b0bec5")
        self.right_panel.grid(row=0, column=1, padx=10, pady=10, sticky="n")

        # Text area for live transcript
        self.label1 = Label(self.right_panel, text="Live Transcript", font=("Arial", 12, "bold"), bg="white")
        self.label1.pack(fill=tk.X, pady=(0, 5))
        self.text_area1 = Text(self.right_panel, width=100, height=19, state='disabled')
        self.text_area1.pack(fill=tk.X, pady=(0, 10))

        # Text area for any other detections (optional)
        self.label2 = Label(self.right_panel, text="Model Detections", font=("Arial", 12, "bold"), bg="white")
        self.label2.pack(fill=tk.X, pady=(0, 5))
        self.text_area2 = Text(self.right_panel, width=100, height=10, state='disabled')
        self.text_area2.pack(fill=tk.X, pady=(0, 10))

        # Button panel
        self.button_panel = Frame(self.main_frame, bg="#b0bec5")
        self.button_panel.grid(row=2, column=1, pady=10, sticky="s")

        # Initially the button says "Start Demo"
        self.start_button = Button(self.button_panel, text="Start Demo", bg="#7749F8", fg="white",
                                   command=self.toggle_transcription)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.export_button = Button(self.button_panel, text="Export Data", bg="#7749F8", fg="white")
        self.export_button.pack(side=tk.LEFT, padx=5)

        # Setup video capture
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.update_video()

        # -----------------------------
        # 2) SETUP AUDIO + TRANSCRIPTION
        # -----------------------------
        self.SAMPLE_RATE = 16000
        # 1-second chunks so we transcribe frequently
        self.CHUNK_DURATION = 1.0
        self.CHANNELS = 1

        # Volume threshold in dBFS.
        # A typical quiet room might be around -60 dBFS.
        # Speaking voice might be -20 to -15 dBFS (or higher).
        self.VOLUME_THRESHOLD_DBFS = -25.0

        # Queues for audio + transcriptions
        self.audio_queue = queue.Queue()
        self.transcription_queue = queue.Queue()

        # Flags & threads
        self.stop_event = threading.Event()
        self.record_thread = None
        self.process_thread = None
        self.transcription_running = False  # Toggle flag

        # On app close, release resources
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Center the window and lock its size
        self.center_window_and_lock()

    def center_window_and_lock(self):
        self.root.update_idletasks()
        width  = self.root.winfo_reqwidth()
        height = self.root.winfo_reqheight()

        screen_width  = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        self.root.geometry(f"+{x}+{y}")

        # Prevent resizing
        self.root.resizable(False, False)

    # -----------------------------
    # 3) WEBCAM LOOP
    # -----------------------------
    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (800, 600))
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

        # Schedule next update in 10 ms
        self.root.after(10, self.update_video)

    # -----------------------------
    # 4) TEXT AREA UTILS
    # -----------------------------
    def add_text_to_text_area1(self, text):
        self.text_area1.config(state='normal')
        self.text_area1.insert(tk.END, text + "\n")
        self.text_area1.see(tk.END)  # auto-scroll to bottom
        self.text_area1.config(state='disabled')

    def add_text_to_text_area2(self, text):
        self.text_area2.config(state='normal')
        self.text_area2.insert(tk.END, text + "\n")
        self.text_area2.see(tk.END)
        self.text_area2.config(state='disabled')

    # -----------------------------
    # 5) BACKGROUND AUDIO THREADS
    # -----------------------------
    def record_audio(self):
        """ Continuously capture audio from mic and push frames to audio_queue. """
        with sd.InputStream(samplerate=self.SAMPLE_RATE, channels=self.CHANNELS, callback=self.audio_callback):
            while not self.stop_event.is_set():
                sd.sleep(100)

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print("Audio status:", status)
        self.audio_queue.put(indata.copy())

    def process_audio(self):
        """
        Pull frames from audio_queue, combine into 1-second chunks,
        check volume via pydub, then transcribe if chunk is above threshold.
        """
        fine_tuned_path = "./whisper-small-en-finetuned"  # Adjust if needed
        device = 0 if torch.cuda.is_available() else -1

        # Minimal pipeline
        asr = pipeline(
            "automatic-speech-recognition",
            model=fine_tuned_path,
            device=device
        )

        frames_per_chunk = int(self.SAMPLE_RATE * self.CHUNK_DURATION)

        # Write a header to the text file (optional)
        with open("LiveTranscription.txt", "a", encoding="utf-8") as f:
            f.write(f"\nLive Transcription for {time.strftime('%Y-%m-%d', time.localtime())}\n")
            f.write("-" * 50 + "\n")

        while not self.stop_event.is_set():
            frames = []
            frames_collected = 0

            # Collect frames until we have a full chunk (1 second)
            while frames_collected < frames_per_chunk and not self.stop_event.is_set():
                data = self.audio_queue.get()
                frames.append(data)
                frames_collected += data.shape[0]

            if self.stop_event.is_set():
                break

            # Combine into one chunk
            audio_chunk = np.concatenate(frames, axis=0).flatten()

            # Convert float32 array -> pydub AudioSegment for dBFS check
            segment = self.float32_to_pydub(audio_chunk, self.SAMPLE_RATE)
            dbfs_val = segment.dBFS

            # Debug print
            print(f"[DEBUG] dBFS: {dbfs_val:.2f}  (threshold={self.VOLUME_THRESHOLD_DBFS})")

            if dbfs_val > self.VOLUME_THRESHOLD_DBFS:
                # It's loud enough to assume speech
                chunk_time_str = time.strftime("%H:%M:%S", time.localtime())
                try:
                    result = asr(audio_chunk)
                except Exception as e:
                    print(f"Error during transcription: {e}")
                    continue

                raw_text = result.get("text", "")
                cleaned_text = self.sanitize_english_text(raw_text).strip()

                if cleaned_text:
                    transcript_line = f"[{chunk_time_str}] {cleaned_text}"
                    print("Transcript:", transcript_line)  # Debug

                    # Write to file
                    with open("LiveTranscription.txt", "a", encoding="utf-8") as f:
                        f.write(transcript_line + "\n")

                    # Send to GUI queue
                    self.transcription_queue.put(transcript_line)
                else:
                    print("Transcript: (empty after cleaning)")
            else:
                # If chunk is too quiet, skip
                print("Transcript: (skipped due to low volume)")

    # -----------------------------
    # 6) HELPER: float32 -> pydub AudioSegment
    # -----------------------------
    def float32_to_pydub(self, float_array, sample_rate=16000):
        """
        Convert a float32 numpy array (range -1.0..1.0) to a pydub AudioSegment.
        This lets us measure dBFS easily (for gating).
        """
        # Ensure float in [-1.0..1.0]
        float_array = np.clip(float_array, -1.0, 1.0)

        # Scale to int16
        int16_array = (float_array * 32767).astype(np.int16)

        # Create pydub AudioSegment from raw PCM data
        segment = AudioSegment(
            data=int16_array.tobytes(),
            sample_width=2,  # 16 bits = 2 bytes
            frame_rate=sample_rate,
            channels=1
        )
        return segment

    # -----------------------------
    # 7) POLLING THE TRANSCRIPTION QUEUE
    # -----------------------------
    def poll_transcriptions(self):
        while not self.transcription_queue.empty():
            line = self.transcription_queue.get()
            self.add_text_to_text_area1(line)

        if self.transcription_running and not self.stop_event.is_set():
            self.root.after(200, self.poll_transcriptions)

    # -----------------------------
    # 8) TOGGLE / START / STOP
    # -----------------------------
    def toggle_transcription(self):
        if not self.transcription_running:
            self.transcription_running = True
            self.start_transcription()
            self.start_button.config(text="Stop Demo")
        else:
            self.stop_transcription()
            self.start_button.config(text="Start Demo")
            self.transcription_running = False

    def start_transcription(self):
        """Spawn the background threads for audio recording + processing."""
        # Clear the text area for a fresh start
        self.text_area1.config(state='normal')
        self.text_area1.delete("1.0", tk.END)
        self.text_area1.config(state='disabled')

        self.stop_event.clear()

        self.record_thread = threading.Thread(target=self.record_audio, daemon=True)
        self.process_thread = threading.Thread(target=self.process_audio, daemon=True)

        self.record_thread.start()
        self.process_thread.start()

        self.poll_transcriptions()

    def stop_transcription(self):
        """Signal threads to stop and wait briefly for them to exit."""
        self.stop_event.set()
        time.sleep(1.0)

    def on_closing(self):
        if self.transcription_running:
            self.stop_transcription()

        self.cap.release()
        self.root.destroy()

    # -----------------------------
    # 9) UTIL: SANITIZE TEXT
    # -----------------------------
    def sanitize_english_text(self, text):
        return re.sub(r"[^A-Za-z0-9\s,.!?'\"]", "", text)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
