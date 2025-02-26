import zipfile
import os
import shutil

def main():
    # Unzip all the folders in the dataset using the base path of the dataset
    base = './_Archive'
    for f in os.listdir(base):
        if f.endswith('.zip'):
            with zipfile.ZipFile(base + '/' + f, 'r') as z:
                z.extractall(base + '/' + f[:-4])
                
    # Go into each folder and move the contents of the audio and
    # transcript folders
    os.mkdir(base + '/audio')
    os.mkdir(base + '/transcripts')
    
    for f in os.listdir(base):
        if os.path.isdir(base + '/' + f) and f != 'audio' and f != 'transcripts' and not f.endswith('.zip'):
            if os.path.exists(base + '/' + f + '/audio'):
                for audio in os.listdir(base + '/' + f + '/audio'):
                    shutil.move(base + '/' + f + '/audio/' + audio, base + '/audio')
                shutil.rmtree(base + '/' + f + '/audio')
            if os.path.exists(base + '/' + f + '/transcripts'):
                for transcript in os.listdir(base + '/' + f + '/transcripts'):
                    shutil.move(base + '/' + f + '/transcripts/' + transcript, base + '/transcripts')
                shutil.rmtree(base + '/' + f + '/transcripts')
            shutil.rmtree(base + '/' + f)


if __name__ == '__main__':
    main()
    base = './_Archive'
