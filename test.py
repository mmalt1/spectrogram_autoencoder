# import subprocess
import os
# def convert_wav_to_mp3(input_wav, output_mp3):
#     try:
#         # Call Sox to convert the file
#         result = subprocess.run(['sox', input_wav, output_mp3], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         print(f"Conversion successful: {input_wav} to {output_mp3}")
#     except subprocess.CalledProcessError as e:
#         print(f"Error occurred: {e.stderr.decode()}")

# # Example usage
# input_wav = '/work/tc062/tc062/s2501147/autoencoder/libritts_data/libritts_dev_clean/wavs/84_121123_000007_000001.wav'
# output_mp3 = '/work/tc062/tc062/s2501147/autoencoder/test_noised_wavs//84_121123_000007_000001.mp3'
# convert_wav_to_mp3(input_wav, output_mp3)

unseen_data_directory = "/work/tc062/tc062/s2501147/autoencoder/aircon1"

for root, dirs, files in os.walk(unseen_data_directory):
    print("Current directory:", root)
    
    # Extract the subdirectory name relative to the main directory
    relative_path = os.path.relpath(root, unseen_data_directory)
    
    # Check if we are not at the root of unseen_data_directory
    if relative_path != ".":
        subdirectory_name = relative_path.split(os.sep)[0]
    else:
        subdirectory_name = None
        
    # Filter and print .pt files along with the subdirectory name
    pt_files = [file for file in files if file.endswith('.pt')]
    for pt_file in pt_files:
        if subdirectory_name:
            print(f"Subdirectory: {subdirectory_name}, File: {pt_file}")
        else:
            print(f"File: {pt_file}")
    
    print()  # Print a blank line for better readability between iterations