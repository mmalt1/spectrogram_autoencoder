# NOTE: txt file necessary for FastPitch preprocessing steps
import os

# Define the paths
wav_directory = '/work/tc062/tc062/s2501147/autoencoder/libritts_data/libritts_r/train/wavs'
output_txt_file = '/work/tc062/tc062/s2501147/autoencoder/libritts_data/libritts_r/train/output.txt'
dummy_transcription = "This is a dummy transcription."

with open(output_txt_file, 'w') as f:
    for root, dirs, files in os.walk(wav_directory):
        for file in files:
            if file.endswith('.wav'):
                relative_path = os.path.relpath(os.path.join(root, file), start=wav_directory)
                
                formatted_path = f'wavs/{relative_path.replace(os.sep, "_")}'
                
                f.write(f'{formatted_path}|{dummy_transcription}\n')
                print(f"File {file} formatted in {output_txt_file}")

print(f'Transcriptions saved to {output_txt_file}')
