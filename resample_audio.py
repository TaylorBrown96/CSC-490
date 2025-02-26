import os
from pydub import AudioSegment

def resample_to_16k(input_dir, output_dir):
    """
    Resample all .wav files in `input_dir` to 16 kHz and save them to `output_dir`.
    """

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        # Skip anything that is not a .wav file
        if not filename.lower().endswith(".wav"):
            continue

        # Construct the full path to the input file
        input_path = os.path.join(input_dir, filename)

        # Load the WAV file using pydub
        audio = AudioSegment.from_wav(input_path)

        # Resample to 16 kHz
        audio_16k = audio.set_frame_rate(16000)

        # Construct the full path for the output file
        output_path = os.path.join(output_dir, filename)

        # Export the resampled file
        audio_16k.export(output_path, format="wav")

        print(f"Resampled: {input_path} -> {output_path}")

def main():
    # Modify these paths as needed
    input_directory = "audio"
    output_directory = "converted"

    resample_to_16k(input_directory, output_directory)
    print("Resampling completed!")

if __name__ == "__main__":
    main()