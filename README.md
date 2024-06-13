# issue
Our issue:
We are passing our audio file to the mel spectrogram and not getting the desired output...

**Thought process:**
- 35.wav is the audio file we are taking as reference
- 35.wav_audio_data.txt is the extracted audio time series we manually input in swift to make sure we start with the right input
- mel_spectrogram_in_python.py is the python code to generate the mel spectrogram
- xyz.py is our python code to export mel_spectrogram_in_python.py into an ml file
- Mel_spectogram.mlpackage-20240611T134525Z-001 (2).zip is the package we get and that is then opened and added to swift
- MelSpectrogramSwift is the folder of swift code, in particular in viewController we load the audio file and the pass it through the mel spectrogram.

- The output we should get if it worked is:
[[-51.309372 -51.109604 -52.23552 -53.613808 -54.30912 ]
 [-62.51481 -62.78988 -64.4379  -65.80139 -64.943535]
 [-57.837097 -58.16116 -60.79122 -64.94867 -66.45402 ]
 [-54.830452 -55.455116 -57.3638  -59.34929 -60.34854 ]
 [-35.4801  -33.566555 -33.08417 -33.642914 -34.932796]]
And have this shape: (40, 173)


