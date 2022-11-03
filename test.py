import os
import numpy.fft
from scipy.io import wavfile
import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
from os.path import join as pjoin
from scipy import signal
from scipy import fft
import math
from matplotlib.widgets import Slider
import pandas as pd

note_string = ['C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B']


def path_to_numpy(file_path):
    wav_file_path = pjoin(os.getcwd(), file_path)
    sampling_rate, audio_signal = wavfile.read(wapv_file_path)
    return sampling_rate, audio_signal


def generate_freq_spectrum(x, sf):
    """
    Derive frequency spectrum of a signal from time domain
    :param x: signal in the time domain
    :param sf: sampling frequency
    :returns frequencies and their content distribution
    """
    x = x - np.average(x)  # zero-centering

    n = len(x)
    k = np.arange(n)
    tarr = n / float(sf)    ####num of sample
    frqarr = k / float(tarr)  # two sides frequency range

    frqarr = frqarr[range(n // 2)]  # one side frequency range

    x = fft.fft(x) / n  # fft computing and normalization
    x = x[range(n // 2)]

    return frqarr, abs(x)


def bandpass_filter(raw_signal, fs):
    numerator, denominator = signal.butter(5, [15, 8000], 'bandpass', fs=fs) #order 5 keep frequencies between 15hz and 8000hz. Diff not visiable
    filtered = signal.lfilter(numerator, denominator, raw_signal)
    return filtered


def frequency_to_note(frequency):
    log_peak_freq = math.log(frequency, 2)
    index = round((log_peak_freq-3.94802101634847)/0.0833334184464846)
    octave_num, note = index // 12, int(index % 12 - 1)
    return note_string[note] + str(octave_num)


def slice_and_find_note(audio_signal, f_rate, section_length, threshold):
    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 5,
            }
    total_duration = len(audio_signal) / f_rate
    n_section, sample_per_section = math.ceil(total_duration / (section_length / 1000)), int((section_length / 1000) * f_rate)
    max_y = numpy.max(audio_signal)
    for section_num in range(0, n_section):
        section = audio_signal[section_num * sample_per_section: min((section_num + 1) * sample_per_section, len(audio_signal))]  # chop into section
        frequency, signal_amplitude = generate_freq_spectrum(section, f_rate)  # fft
        peak_frequency_index = numpy.argmax(signal_amplitude)  # get peak freq, return the index of the highest value
        if signal_amplitude[peak_frequency_index] > threshold:
            plt.text(section_num * sample_per_section, 0, frequency_to_note(frequency[peak_frequency_index]), fontdict= font)  # plot note name
            plt.text(section_num * sample_per_section, 0.25*max_y, frequency[peak_frequency_index], fontdict= font)  # freq
        plt.axvline(x=min((section_num + 1) * sample_per_section, len(audio_signal)), color='r', linewidth=0.5, linestyle="-",zorder=10)  # lines for separating segments
        plt.text(section_num * sample_per_section, 0.5*max_y, round(signal_amplitude[peak_frequency_index]), fontdict= font)  # plot the freq magnitude

    plt.plot(audio_signal, zorder=0)
    plt.show()
    #the lower number is freq higher is mag


def to_do():
    # heuristic

    # interactive plot
    #a record
    #a full a process, b record
    #b full b process, a record
    pass


def realtime_vanilla(time, fs, frame_time):

    concatenated_audio = np.array([])
    for i in range(0, time * fs, int(frame_time * fs)):
        recording_segment = sd.rec(int(frame_time * fs), samplerate=fs, channels=1)
        sd.wait()
        recording_segment = recording_segment.reshape((len(recording_segment),))
        concatenated_audio = np.concatenate([concatenated_audio, recording_segment])
        filtered_segment = bandpass_filter(recording_segment, fs)
        freq, signal_amp = generate_freq_spectrum(filtered_segment, fs)  # fft
        peak_freq_index = numpy.argmax(signal_amp)  # get peak freq, return the index of the highest value
        note_name = frequency_to_note(freq[peak_freq_index])
        print(note_name)
    sd.wait()
    return concatenated_audio

def slice_and_find_note_interactive(audio_signal, f_rate, section_length, threshold):
    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 5,
            }
    fig, (ax1, ax2) = plt.subplots(2)
    total_duration = len(audio_signal) / f_rate
    n_section, sample_per_section = math.ceil(total_duration / (section_length / 1000)), int((section_length / 1000) * f_rate)
    max_y = numpy.max(audio_signal)
    for section_num in range(0, n_section):
        section = audio_signal[section_num * sample_per_section: min((section_num + 1) * sample_per_section, len(audio_signal))]  # chop into section
        frequency, signal_amplitude = generate_freq_spectrum(section, f_rate)  # fft
        peak_frequency_index = numpy.argmax(signal_amplitude)  # get peak freq, return the index of the highest value
        if signal_amplitude[peak_frequency_index] > threshold:
            ax1.text(section_num * sample_per_section, 0, frequency_to_note(frequency[peak_frequency_index]), fontdict= font)  # plot note name
            ax1.text(section_num * sample_per_section, 0.25*max_y, frequency[peak_frequency_index], fontdict= font)  # freq
        ax1.axvline(x=min((section_num + 1) * sample_per_section, len(audio_signal)), color='r', linewidth=0.5, linestyle="-",zorder=10)  # lines for separating segments
        ax1.text(section_num * sample_per_section, 0.5*max_y, round(signal_amplitude[peak_frequency_index]), fontdict= font)  # plot the freq magnitude
    ax1.plot(audio_signal, zorder=0)
    axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    fig.subplots_adjust(left=0.25, bottom=0.25)
    freq_slider = Slider(
        ax=axfreq,
        label='Frequency [Hz]',
        valmin=0.1,
        valmax=30,
        valinit=5,
    )

    ax1.set_xlabel('Sample number')
    ax1.set_ylabel('Magnitude')
    ax2.set_xlabel('Freq (Hz)')
    ax2.set_ylabel('Magnitude')
    plt.show()



# register the update function with each slider

if __name__ == "__main__":
    print("enter mode number: 1 for real, 2 for record, 3 for file")
    mode = int(input())

    if mode == 1:  # note mapping is wrong
        fs = 96000
        audio = realtime_vanilla(10, fs, 0.5)
        print(audio.shape)
        plt.plot(audio)
        plt.show()

    if mode == 2:
        fs = 48000
        duration = 10
        frame_size = 500  # length of each section in ms
        magnitude_threshold = 0 # need to change this to value based on average. find normal not average
        recording_signal = sd.rec(int(duration * fs), samplerate=fs, channels=1) #return type numpy.ndarry shape (n,1)
        sd.wait()
        recording_signal = recording_signal.reshape((len(recording_signal),))
        filtered_signal = bandpass_filter(recording_signal, fs)
        slice_and_find_note(filtered_signal, fs, frame_size, magnitude_threshold)

    if mode == 3:
        frame_size = 500  # length of each section in ms
        magnitude_threshold = 50
        path = "recording.wav"
        frame_rate, signal_numpy = path_to_numpy(path) #signal_numpy shape (n,)
        filtered_signal = bandpass_filter(signal_numpy, frame_rate)
        slice_and_find_note(filtered_signal, frame_rate, frame_size, magnitude_threshold)

    if mode == 4:
        df = pd.read_csv("data.txt")
        print(len(df))
        df = df.to_numpy().reshape(len(df),)
        print(df.shape)
        frame_size = 500  # length of each section in ms
        magnitude_threshold = 0
        frame_rate = 48000

        filtered_signal = bandpass_filter(df, frame_rate)
        slice_and_find_note(filtered_signal, frame_rate, frame_size, magnitude_threshold)

    if mode == 5:
        while(True):
            print("type 1 to connect")
            mode = int(input())
            if(mode== 1):
                path = "Capstone piano test 1.wav"
                frame_rate, signal_numpy = path_to_numpy(path)  # signal_numpy shape (n,)
                filtered_signal = bandpass_filter(signal_numpy, frame_rate)
                print("enter value 1")
                f1 = int(input())
                print("enter value 2")
                f2 = int(input())
                freq, magnitude = generate_freq_spectrum(filtered_signal[f1:f2], frame_rate)
                plt.plot(freq, magnitude)
                plt.show()

    if mode == 6: #interactive
        pass
#1.5
