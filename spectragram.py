from scipy.io import wavfile
import os
from os.path import join as pjoin
import matplotlib.pyplot as plt

if __name__ == "__main__":
    wav_file_path = pjoin(os.getcwd(), "test c4c5.wav")
    sampling_rate, audio_signal = wavfile.read(wav_file_path)
    plt.subplot(211)
    plt.title('Spectrogram of a wav file with piano music')
    plt.plot(audio_signal[5000:300000])
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.subplot(212)
    plt.specgram(audio_signal[5000:300000], Fs=sampling_rate)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()


wav_file_path = pjoin(os.getcwd(), "test c4c5.wav")
sampling_rate, audio_signal = wavfile.read(wav_file_path)
