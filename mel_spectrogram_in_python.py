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
