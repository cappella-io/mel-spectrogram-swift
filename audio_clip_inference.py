import librosa
import torch
import torch.nn as nn
import numpy as np

class MelSpectrogram(nn.Module):
    def __init__(self, sample_rate, fft_size=2048, hop_length=255, num_mels=40):
        super(MelSpectrogram, self).__init__()
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.num_mels = num_mels

    def forward(self, x):
        print("Converting audio to Mel Spectrogram")
        mel_spec = librosa.feature.melspectrogram(y=x.numpy().flatten(),
                                                  sr=self.sample_rate,
                                                  n_fft=self.fft_size,
                                                  hop_length=self.hop_length,
                                                  n_mels=self.num_mels)
        print(f"Mel Spectrogram shape: {mel_spec.shape}")
        print("Mel Spectrogram (first 5 samples):")
        print(mel_spec[:5, :5])  # Print the first 5x5 values of the Mel spectrogram
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        print(f"Mel Spectrogram (dB) shape: {mel_spec.shape}")
        print("Mel Spectrogram (dB, first 5 samples):")
        print(mel_spec[:5, :5])  # Print the first 5x5 values of the Mel spectrogram in dB
        return torch.Tensor(mel_spec).unsqueeze(0)  # Add batch dimension

def get_featurizer(sample_rate):
    print("Creating MelSpectrogram featurizer")
    return MelSpectrogram(sample_rate=sample_rate)

def load_audio_file(file_path, sample_rate):
    print(f"Loading audio file from: {file_path}")
    data, _ = librosa.load(file_path, sr=sample_rate)
    #data, _ = librosa.load(file_path, sr=45504/2)
    #data, _ = librosa.load(file_path, sr=None, mono=False)
    
    # Separate the channels
    #left_channel = data[0]
    #right_channel = data[1]
    #print(f"Left Channel audio data shape: {left_channel.shape}")
    #print(f"Right Channel audio data shape: {right_channel.shape}")

    # Save the raw data to binary files
#    left_channel_bytes = left_channel.tofile('left_channel_raw_python.bin')
#    right_channel_bytes = right_channel.tofile('right_channel_raw_python.bin')

    #print("Original audio data (first 5 samples):")
    #for i in (range(0,5)):
        #print(f"left_channel[{i}]: {data[i]}")
        #print(f"left_channel[{i}]: {left_channel[i]}")
        #print(f"right_channel[{i}]: {right_channel[i]}")
        #print(f"average[{i}]: {(right_channel[i]+left_channel[i])/2}")
    #print(f"Original audio data shape: {data.shape}")

    #max_val = np.max(np.abs(data))
    #print(f"Maximum value in original audio data: {max_val}")

    #data = data.astype(np.float32)
    #data = data / max_val

    #print("Normalized audio data (first 5 samples):")
    #for i, value in enumerate(data[:5]):
    #    print(f"y[{i}]: {value}")
    #print(f"Normalized audio data shape: {data.shape}")

    print (data)
    # Save the array to a text file
    #np.savetxt('audio_data.txt', data)
    
    # Convert the array to a list of strings
    audio_data_str = [str(num) for num in data]

# Join the list into a single string with commas
    audio_data_str = ','.join(audio_data_str)

# Write the string to a text file
    output_file = 'audio_data.txt'  # replace with your desired output file path
    with open(output_file, 'w') as file:
        file.write(audio_data_str)

    print("Audio data has been exported to", output_file)
    
    #data.astype(np.float32).tofile('audio_data.bin')

    return data


def compare_binary_files(file1, file2):
    with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
        data1 = f1.read()
        data2 = f2.read()
        
        if data1 == data2:
            print("Files are identical")
        else:
            print("Files are different")

# Compare left channels
#compare_binary_files('left_channel_raw_python.bin', 'left_channel_raw_swift.bin')

# Compare right channels
#compare_binary_files('right_channel_raw_python.bin', 'right_channel_raw_swift.bin')


def run_model_on_audio(model_file, audio_file, sample_rate=22050, record_seconds=2):
    # Load the model
    print(f"Loading model from: {model_file}")
    model = torch.jit.load(model_file)
    model.eval().to('cpu')
    print("Model loaded and set to evaluation mode")
    
    # Load and process audio file
    audio_data = load_audio_file(audio_file, sample_rate)
    expected_length = sample_rate * record_seconds
    print(f"Expected audio data length: {expected_length}")
    if len(audio_data) != expected_length:
        print(f"Padding or truncating audio data from length {len(audio_data)} to {expected_length}")
        audio_data = np.pad(audio_data, (0, max(0, expected_length - len(audio_data))), 'constant')[:expected_length]
    print(f"Processed audio data shape: {audio_data.shape}")
    
    # Convert audio data to tensor and compute mel spectrogram
    waveform_tensor = torch.tensor(audio_data).unsqueeze(0).unsqueeze(0)
    print(f"Waveform tensor shape: {waveform_tensor.shape}")
    featurizer = get_featurizer(sample_rate)
    mel_spectrogram = featurizer(waveform_tensor)
    print(f"Mel spectrogram shape before adding channel dimension: {mel_spectrogram.shape}")
    mel_spectrogram = mel_spectrogram.unsqueeze(1)  # Add channel dimension
    print(f"Mel spectrogram shape after adding channel dimension: {mel_spectrogram.shape}")

    # Print what the model takes as input
    print("Model input (first 5x5 samples of the first channel):")
    print(mel_spectrogram[0, 0, :5, :5])  # Print the first 5x5 values of the input to the model    
    
    # Run the model
    print("Running model on the Mel spectrogram")
    with torch.no_grad():
        out = model(mel_spectrogram)
        print(f"Raw model output: {out}")
        prediction = torch.round(torch.sigmoid(out)).item()
        print(f"Sigmoid output: {torch.sigmoid(out)}, Rounded prediction: {prediction}")
    
    return prediction

# Define file paths and run the model
model_file = 'encoded_CNNLSTM_new_valid_model_trace.pt'
audio_file = r"C:\Users\ApollineDeroche\Downloads\35.wav"
print("Starting the process...")
prediction = run_model_on_audio(model_file, audio_file)
print("Final Prediction:", prediction)
