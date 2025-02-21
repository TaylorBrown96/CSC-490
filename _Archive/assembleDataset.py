import zipfile
import os
import shutil

def main():
    # Unzip all the folders in the dataset
    for f in os.listdir('.'):
        if f.endswith('.zip'):
            with zipfile.ZipFile(f, 'r') as z:
                z.extractall(f[:-4])
                
    # Go into each folder and move the contents of the audio and 
    # transcript folders to the main folder
    os.mkdir('audio')
    os.mkdir('transcripts')
    
    for f in os.listdir('.'):
        if os.path.isdir(f) and f != 'audio' and f != 'transcripts' and f != f.endswith('.zip'):
            if os.path.exists(f + '/audio'):
                for audio in os.listdir(f + '/audio'):
                    shutil.move(f + '/audio/' + audio, 'audio')
                shutil.rmtree(f + '/audio')
            if os.path.exists(f + '/transcripts'):
                for transcript in os.listdir(f + '/transcripts'):
                    shutil.move(f + '/transcripts/' + transcript, 'transcripts')
                shutil.rmtree(f + '/transcripts')
            shutil.rmtree(f)
            

if __name__ == '__main__':
    main()
