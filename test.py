# import subprocess

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


noised_wav = "/work/tc062/tc062/s2501147/FastPitch/FastPitches/PyTorch/SpeechSynthesis/FastPitch/wav_files/mels/noisy_audio2.pt"
noised_spec = "/work/tc062/tc062/s2501147/FastPitch/FastPitches/PyTorch/SpeechSynthesis/FastPitch/torch_saved/mels/speaker_1.pt"