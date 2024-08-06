import wave
import lameenc

def convert_wav_to_mp3_lame(wav_file_path, mp3_file_path, bit_rate=192):
    # Open the WAV file
    with wave.open(wav_file_path, 'rb') as wav_file:
        # Read the WAV file properties
        n_channels = wav_file.getnchannels()
        sampwidth = wav_file.getsampwidth()
        framerate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        
        # Read the audio frames
        audio_frames = wav_file.readframes(n_frames)
        
    # Create an encoder instance
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bit_rate)
    encoder.set_in_sample_rate(framerate)
    encoder.set_channels(n_channels)
    encoder.set_quality(2)  # 2=high, 5 = medium, 7=low
    
    # Encode the audio to MP3 format
    mp3_data = encoder.encode(audio_frames)
    mp3_data += encoder.flush()
    
    # Write the MP3 data to the output file
    with open(mp3_file_path, 'wb') as mp3_file:
        mp3_file.write(mp3_data)
    
    print(f"Converted {wav_file_path} to {mp3_file_path} with bit rate {bit_rate} kbps")

# Example usage
wav_file_path = '/work/tc062/tc062/s2501147/autoencoder/libritts_data/libritts_dev_clean/wavs/84_121123_000007_000001.wav'
mp3_file_path = '/work/tc062/tc062/s2501147/autoencoder/mp3_data/84_121123_000007_000001.mp3'
bit_rate = 192  # Change this to 128, 256, etc., for different bit rates

convert_wav_to_mp3_lame(wav_file_path, mp3_file_path, bit_rate)
