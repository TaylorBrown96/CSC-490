import tkinter as tk
import pyaudio
import wave
import threading
import os

Sentences = [
    "I have placed a tourniquet on the upper arm and the bleeding is under control.",
    "The tourniquet is secured on the thigh and no further blood loss is observed.",
    "I applied a tourniquet to the left leg and the bleeding has stopped.",
    "A tourniquet has been placed on the right forearm and circulation is restricted.",
    "I’ve got a tourniquet on the wounded limb and the hemorrhaging has ceased.",
    "Tourniquet applied to the upper leg and it is holding firm.",
    "The tourniquet is secured above the wound and the blood flow has been cut off.",
    "I placed a tourniquet on the left arm and checked for a distal pulse.",
    "A tourniquet is in place on the thigh and it is properly tightened.",
    "I’ve applied a tourniquet to the injured limb and the bleeding is no longer active.",
    "Tourniquet applied on the right arm and pressure is sufficient.",
    "A tourniquet is locked in place on the leg and the bleeding has been controlled.",
    "I secured the tourniquet above the knee and blood loss has stopped.",
    "The injured arm has a tourniquet and circulation is fully restricted.",
    "Tourniquet applied to the upper limb and no further blood loss is occurring.",
    "A tourniquet is holding on the right leg and it is effectively stopping the bleeding.",
    "I tightened a tourniquet on the left thigh and it is working as intended.",
    "The tourniquet is in position on the forearm and bleeding is minimal now.",
    "I’ve applied a tourniquet to the leg and checked that it is properly secured.",
    "The limb is secured with a tourniquet and no excessive bleeding is present.",
    "Tourniquet is applied on the upper arm and I have verified proper pressure.",
    "A tourniquet is in place on the thigh and I have confirmed there is no distal pulse.",
    "The tourniquet is fastened on the forearm and the bleeding is controlled.",
    "I applied a tourniquet to the upper thigh and checked for proper placement.",
    "Tourniquet applied to the leg and the casualty is stable.",
    "The wounded arm has a tourniquet and I have monitored for circulation return.",
    "I placed a tourniquet above the wound and ensured it was high and tight.",
    "A tourniquet has been applied to the upper limb and pressure is stable.",
    "The tourniquet is properly secured on the right arm and blood loss has ceased.",
    "I’ve placed a tourniquet above the injury site and checked for effectiveness.",
    "Tourniquet is on the leg and the casualty’s bleeding is under control.",
    "A tourniquet has been placed on the left forearm and I confirmed full restriction.",
    "I applied a tourniquet to the upper right thigh and the hemorrhage has stopped.",
    "The injured limb has been stabilized with a tourniquet and bleeding is controlled.",
    "I secured a tourniquet to the left leg and there is no distal circulation.",
    "The tourniquet is in place and the casualty is no longer losing blood.",
    "A tourniquet has been applied to the wounded limb and it is functioning properly.",
    "I tightened a tourniquet above the knee and bleeding is contained.",
    "The forearm is secured with a tourniquet and circulation is cut off.",
    "Tourniquet is locked onto the left thigh and bleeding has stopped.",
    "I have placed a tourniquet on the right arm and checked for a pulse.",
    "A tourniquet is in place on the lower limb and the wound is stabilized.",
    "The wound site is secured with a tourniquet and no further blood loss is observed.",
    "Tourniquet applied on the upper right arm and bleeding is under control.",
    "I have positioned a tourniquet on the left thigh and verified pressure.",
    "A tourniquet has been properly secured on the wounded leg.",
    "The tourniquet is placed high on the arm and bleeding is stopped.",
    "Tourniquet applied to the forearm and no further hemorrhaging is occurring.",
    "I tightened a tourniquet on the leg and it is working effectively.",
    "The casualty has a tourniquet on the upper limb and no circulation is detected.",
    "I applied a tourniquet to the lower limb and confirmed full occlusion.",
    "A tourniquet is in place on the leg and the casualty is stable.",
    "I secured a tourniquet on the injured arm and verified that it is effective.",
    "Tourniquet applied to the thigh and the bleeding has completely stopped.",
    "A tourniquet has been placed above the injury and the casualty is no longer at risk of excessive bleeding.",
    "The right leg has a tourniquet and it is preventing further blood loss.",
    "I have applied a tourniquet to the left arm and ensured proper pressure.",
    "The casualty’s limb is secured with a tourniquet and I confirmed it is working.",
    "Tourniquet applied above the knee and I have checked for circulation.",
    "The upper arm is secured with a tourniquet and hemorrhaging is controlled.",
    "A tourniquet is in place on the thigh and I have confirmed full restriction.",
    "I placed a tourniquet above the wound and checked for a pulse.",
    "The casualty has a tourniquet on the lower limb and the bleeding is controlled.",
    "I secured a tourniquet to the right forearm and monitored for effectiveness.",
    "Tourniquet applied on the upper leg and no further bleeding is present.",
    "A tourniquet has been locked onto the left arm and the blood loss is stopped.",
    "The casualty’s forearm is stabilized with a tourniquet and there is no distal pulse.",
    "I placed a tourniquet on the injured limb and confirmed proper tightness.",
    "Tourniquet applied to the leg and bleeding has been successfully controlled.",
    "A tourniquet has been placed above the elbow and hemorrhaging has ceased.",
    "The casualty has a tourniquet on the thigh and it is stopping the blood flow.",
    "I secured a tourniquet on the lower limb and confirmed it is functioning.",
    "The arm has a tourniquet and I verified no return of circulation.",
    "Tourniquet applied above the wound and I checked for effectiveness.",
    "The leg is stabilized with a tourniquet and the casualty is stable.",
    "A tourniquet has been secured on the right thigh and the bleeding is under control.",
    "The tourniquet is in place on the upper arm and the casualty is no longer losing blood.",
    "I applied a tourniquet to the wounded leg and confirmed no distal pulse.",
    "A tourniquet has been tightened on the left arm and the hemorrhage has stopped.",
    "The injured limb is secured with a tourniquet and I verified its effectiveness.",
    "Tourniquet applied on the upper thigh and bleeding has completely stopped.",
    "I placed a tourniquet on the lower limb and ensured full restriction of blood flow.",
    "A tourniquet has been locked onto the right forearm and no circulation is detected.",
    "The casualty’s thigh has a tourniquet and it is effectively stopping the bleeding.",
    "I applied a tourniquet to the wounded arm and checked for proper pressure.",
    "Tourniquet secured on the leg and the casualty is stable with no further bleeding.",
    "A tourniquet is in place above the wound and the blood loss has been controlled.",
    "The upper arm is secured with a tourniquet and circulation has been fully restricted.",
    "I placed a tourniquet on the injured leg and confirmed proper tightness.",
    "Tourniquet applied to the lower arm and the hemorrhage has ceased.",
    "A tourniquet has been tightened above the knee and bleeding has stopped.",
    "The casualty has a tourniquet on the upper thigh and blood loss is no longer occurring.",
    "I secured a tourniquet on the left arm and verified that it is effective.",
    "The right leg is stabilized with a tourniquet and no further blood loss is present.",
    "A tourniquet is in place on the forearm and it is stopping the bleeding.",
    "Tourniquet applied above the elbow and the casualty’s circulation is restricted.",
    "I placed a tourniquet on the thigh and ensured it was high and tight.",
    "A tourniquet has been secured on the arm and blood flow has been successfully stopped.",
    "The casualty’s wound is stabilized with a tourniquet and bleeding is no longer active.",
    "I applied a tourniquet to the upper leg and confirmed no return of circulation."
]

# Audio Settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

# Users name
NAME = ""

# Make audio subdirectory
if not os.path.exists("audio"):
    os.makedirs("audio")
    
# Make transcripts subdirectory
if not os.path.exists("transcripts"):
    os.makedirs("transcripts")

class AudioRecorder:
    # Initialize the recorder
    def __init__(self):
        self.p = None
        self.stream = None
        self.frames = []
        self.is_recording = False
        self.file_index = 0  # You can use the current sentence index or a separate counter

    # Start recording audio and save to output_filename upon stop
    def start_recording(self, output_filename):
        """Start recording audio and save to output_filename upon stop."""
        self.output_filename = f'audio/{NAME}_{output_filename}'
        self.is_recording = True
        self.frames = []

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT,
                                  channels=CHANNELS,
                                  rate=RATE,
                                  input=True,
                                  frames_per_buffer=CHUNK)

        # Record in a separate thread so it doesn't block the GUI
        self.thread = threading.Thread(target=self._record)
        self.thread.start()

    # Internal loop that keeps pulling audio data until stopped
    def _record(self):
        """Internal loop that keeps pulling audio data until stopped."""
        while self.is_recording:
            data = self.stream.read(CHUNK)
            self.frames.append(data)

    # Stop recording and save the audio to a WAV file
    def stop_recording(self):
        """Stop recording and save the audio to a WAV file."""
        if not self.is_recording:
            return

        self.is_recording = False
        # Wait for the recording thread to finish
        self.thread.join()

        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

        wf = wave.open(self.output_filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        
        # Save transcript
        with open(f"transcripts/{self.output_filename.split('/')[-1].replace('.wav', '.txt')}", "w") as f:
            f.write(Sentences[app.current_index])

# GUI
class App:
    # Initialize the GUI and set up the UI elements
    def __init__(self, root):
        self.root = root
        self.root.title("Tourniquet Recording")

        self.sentences = Sentences
        self.current_index = 0

        self.recorder = AudioRecorder()

        # Create UI elements
        self.sentence_label = tk.Label(root, text=self.sentences[self.current_index], wraplength=850, font=("Arial", 14))
        self.sentence_label.pack(pady=20)

        self.button_frame = tk.Frame(root)
        self.button_frame.pack(pady=10)

        self.record_button = tk.Button(self.button_frame, text="Record", command=self.toggle_record, background="Red")
        self.record_button.pack(side=tk.LEFT, padx=5)

        self.next_button = tk.Button(self.button_frame, text="Next", command=self.next_sentence)
        self.next_button.pack(side=tk.LEFT, padx=5)

    # Handle the start/stop logic of audio recording
    def toggle_record(self):
        """Handle the start/stop logic of audio recording."""
        if not self.recorder.is_recording:
            # Start recording
            self.record_button.config(text="Stop", background="lightgreen")
            filename = f"Sentence_{self.current_index}.wav"
            self.recorder.start_recording(filename)
        else:
            # Stop recording
            self.record_button.config(text="Record", background="Red")
            self.recorder.stop_recording()

    # Move to the next sentence in the list
    def next_sentence(self):
        """Move to the next sentence in the list."""
        # If we are still recording, we should stop first
        if self.recorder.is_recording:
            self.toggle_record()

        self.current_index += 1
        if self.current_index >= len(self.sentences):
            self.current_index = 0  # or handle it differently if you don't want to loop

        self.sentence_label.config(text=self.sentences[self.current_index])


if __name__ == "__main__":
    # Get the user's name before starting the GUI app
    print("What is your name?")
    NAME = input()
    
    root = tk.Tk()
    app = App(root)
    root.mainloop()
