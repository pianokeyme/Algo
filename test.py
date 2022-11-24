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
from scipy.signal import find_peaks

note_string = ['C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B']


def path_to_numpy(file_path):
    wav_file_path = pjoin(os.getcwd(), file_path)
    sampling_rate, audio_signal = wavfile.read(wav_file_path)
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
    tarr = n / float(sf)  ####num of sample
    frqarr = k / float(tarr)  # two sides frequency range
    frqarr = frqarr[range(n // 2)]  # one side frequency range
    x = fft.fft(x) / n  # fft computing and normalization
    x = x[range(n // 2)]
    return frqarr, abs(x)


def autocorr(time_domain_signal):
    convo_result = signal.fftconvolve(time_domain_signal, time_domain_signal)  # np.correlate
    sliced_convo_result = convo_result[
                          convo_result.size // 2:]  # convo result is generally symmetrical so its sliced down the middle
    return sliced_convo_result


def autocorr_freq(time_domain_signal, sampling_rate):  # return fundamental freq found using autocorr
    convo_result = signal.fftconvolve(time_domain_signal, time_domain_signal)  # np.correlate
    ac = convo_result[convo_result.size // 2:]  # len(result)/2 or result.size//2
    phase_zero_height = np.max(ac)  # highest peak value
    if phase_zero_height > 10000000:  # 10M magnitude threshhold
        peaks = find_peaks(ac, height=phase_zero_height / 4, distance=11)[
            0]  # find peaks return a tuple. but second item is empty
        dict = {}  # key is index
        for i in range(len(peaks)):
            peak_index = peaks[i]
            dict[ac[peak_index]] = peak_index
        sorted_value = sorted(dict.keys(), reverse=True)
        freq_from_corr = sampling_rate / abs(dict[sorted_value[0]] - dict[sorted_value[1]])
        return freq_from_corr
    return 0


def bandpass_filter(raw_signal, fs):
    numerator, denominator = signal.butter(5, [20, 4200], 'bandpass',
                                           fs=fs)  # order 5 keep frequencies between 15hz and 8000hz. Diff not visiable
    filtered = signal.lfilter(numerator, denominator, raw_signal)
    return filtered


def frequency_to_note(frequency):
    log_peak_freq = math.log(frequency, 2)
    index = round((log_peak_freq - 3.94802101634847) / 0.0833334184464846)
    octave_num, note = index // 12, int(index % 12 - 1)
    return note_string[note] + str(octave_num)


def slice_and_find_note(audio_signal, f_rate, section_length, threshold):
    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 5,
            }
    total_duration = len(audio_signal) / f_rate
    n_section, sample_per_section = math.ceil(total_duration / (section_length / 1000)), int(
        (section_length / 1000) * f_rate)
    max_y = numpy.max(audio_signal)
    for section_num in range(0, n_section):
        section = audio_signal[section_num * sample_per_section: min((section_num + 1) * sample_per_section,
                                                                     len(audio_signal))]  # chop into section
        frequency, signal_amplitude = generate_freq_spectrum(section, f_rate)  # fft
        peak_frequency_index = numpy.argmax(signal_amplitude)  # get peak freq, return the index of the highest value
        if signal_amplitude[peak_frequency_index] > threshold:
            plt.text(section_num * sample_per_section, 0, frequency_to_note(frequency[peak_frequency_index]),
                     fontdict=font)  # plot note name
            plt.text(section_num * sample_per_section, 0.25 * max_y, math.ceil(frequency[peak_frequency_index]),
                     fontdict=font)  # freq
        plt.axvline(x=min((section_num + 1) * sample_per_section, len(audio_signal)), color='r', linewidth=0.5,
                    linestyle="-", zorder=10)  # lines for separating segments
        plt.text(section_num * sample_per_section, 0.5 * max_y, round(signal_amplitude[peak_frequency_index]),
                 fontdict=font)  # plot the freq magnitude

    plt.plot(audio_signal, zorder=0)
    plt.show()
    # the lower number is freq higher is mag


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


def slice_and_find_note_autocorr(audio_signal, f_rate, section_length):  # now uses autocorr
    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 5,
            }
    fig, (ax1, ax2) = plt.subplots(2)
    total_duration = len(audio_signal) / f_rate
    n_section, sample_per_section = math.ceil(total_duration / (section_length / 1000)), int(
        (section_length / 1000) * f_rate)
    max_y = numpy.max(audio_signal)
    for section_num in range(0, n_section):
        section = audio_signal[section_num * sample_per_section: min((section_num + 1) * sample_per_section,
                                                                     len(audio_signal))]  # chop into section
        frequency = autocorr_freq(section, f_rate)
        if frequency > 0:
            ax1.text(section_num * sample_per_section, 0, frequency_to_note(frequency), fontdict=font)  # plot note name
            ax1.text(section_num * sample_per_section, 0.25 * max_y, frequency, fontdict=font)  # freq
        ax1.axvline(x=min((section_num + 1) * sample_per_section, len(audio_signal)), color='r', linewidth=0.5,
                    linestyle="-", zorder=10)  # lines for separating segments
    ax1.plot(audio_signal, zorder=0)
    ax1.set_xlabel('Sample number')
    ax1.set_ylabel('Magnitude')
    ax2.set_xlabel('Freq (Hz)')
    ax2.set_ylabel('Magnitude')
    plt.show()


# register the update function with each slider

if __name__ == "__main__":
    print("enter mode number: 1 for real, 3 for file")
    mode = int(input())

    if mode == 1:  # note mapping is wrong
        fs = 48000
        audio = realtime_vanilla(10, fs, 0.5)
        print(audio.shape)
        plt.plot(audio)
        plt.show()

    if mode == 2:
        frame_size = 200  # length of each section in ms
        path = "test c4c5.wav"
        frame_rate, signal_numpy = path_to_numpy(path)  # signal_numpy shape (n,)
        filtered_signal = bandpass_filter(signal_numpy, frame_rate)
        slice_and_find_note_autocorr(signal_numpy, frame_rate, frame_size)
