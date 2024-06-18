import librosa
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct

class MelSpectrogram(nn.Module):
    def __init__(self, sample_rate, fft_size=2048, hop_length=255, num_mels=40):
        super(MelSpectrogram, self).__init__()
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.num_mels = num_mels

    def forward(self, x):
        # Using librosa to compute the mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=x.numpy().flatten(),
                                                  sr=self.sample_rate,
                                                  n_fft=self.fft_size,
                                                  hop_length=self.hop_length,
                                                  n_mels=self.num_mels)
        # Convert to dB (optional)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return torch.Tensor(mel_spec).unsqueeze(0)  # Add batch dimension

def get_featurizer(sample_rate):
    return MelSpectrogram(sample_rate=sample_rate)


melspectogram_sample_input = torch.randn(1,44100)  # This is correctly shaped

Melspectogram_MODEL = MelSpectrogram(sample_rate = 22050)

Melspectogram_MODEL.eval()  # Set the model to evaluation mode

traced_model = torch.jit.trace(Melspectogram_MODEL, melspectogram_sample_input)

mlmodel = ct.convert(traced_model, inputs=[ct.TensorType(name="audio", shape=(1,44100))],source="pytorch")

# Save the CoreML model
mlmodel.save("/content/drive/MyDrive/Eval_Mel_spectogram.mlpackage")