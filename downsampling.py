import librosa
import soundfile as sf

# Load the audio file
file_path = '/work/tc062/tc062/s2501147/autoencoder/noise_train/Typing_4.wav'
y, sr = librosa.load(file_path, sr=None)  # Load at original 24000 Hz

# Resample to 22050 Hz
y_resampled = librosa.resample(y, orig_sr=sr, target_sr=22050)

# Save the resampled audio
output_path = '/work/tc062/tc062/s2501147/FastPitch/FastPitches/PyTorch/SpeechSynthesis/FastPitch/wav_files/wavs/Typing_4.wav'
sf.write(output_path, y_resampled, 22050)