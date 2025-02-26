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
    
    
    """# Unzip all the folders in the dataset
    for f in os.listdir('./_Archive'):
        if f.endswith('.zip'):
            with zipfile.ZipFile(f, 'r') as z:
                z.extractall(f[:-4])
                
    # Go into each folder and move the contents of the audio and 
    # transcript folders to the main folder
    os.mkdir('audio')
    os.mkdir('transcripts')
    
    for f in os.listdir('./_Archive'):
        if os.path.isdir(f) and f != 'audio' and f != 'transcripts' and f != f.endswith('.zip'):
            if os.path.exists(f + '/audio'):
                for audio in os.listdir(f + '/audio'):
                    shutil.move(f + '/audio/' + audio, 'audio')
                shutil.rmtree(f + '/audio')
            if os.path.exists(f + '/transcripts'):
                for transcript in os.listdir(f + '/transcripts'):
                    shutil.move(f + '/transcripts/' + transcript, 'transcripts')
                shutil.rmtree(f + '/transcripts')
            shutil.rmtree(f)"""
            

if __name__ == '__main__':
    main()
    base = './_Archive'
