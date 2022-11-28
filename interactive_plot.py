import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.signal import find_peaks
from matplotlib.widgets import Slider
from test import bandpass_filter, generate_freq_spectrum, frequency_to_note, path_to_numpy, autocorr, autocorr_freq

note_string = ['C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B']


class InteractivePlot:

    def __init__(self, audio_signal_array, sampling_rate, frame_size, threshold):
        self.sampling_rate = sampling_rate
        # self.audio_signal_array_filtered = bandpass_filter(audio_signal_array, self.sampling_rate)
        self.audio_signal_array = audio_signal_array
        self.frame_size = frame_size  # in ms
        self.threshold = threshold
        self.signal_length = len(self.audio_signal_array)
        self.sample_per_section = int((self.frame_size / 1000) * self.sampling_rate)
        # function check and replot
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, figsize=(5, 8))
        self.font = {'family': 'serif',
                     'color': 'darkred',
                     'weight': 'normal',
                     'size': 8,
                     }
        axfreq = self.fig.add_axes([0.25, 0.1, 0.65, 0.03])
        self.fig.subplots_adjust(left=0.25, bottom=0.25)
        total_duration = self.signal_length / self.sampling_rate
        n_section = math.ceil(total_duration / (self.frame_size / 1000))
        self.frame_slider = Slider(
            ax=axfreq,
            label='Frame number ',
            valmin=1,
            valmax=n_section,
            valinit=1,
            valstep=1,
        )

        self.ax1.set_xlabel('Sample number')
        self.ax1.set_ylabel('Magnitude')
        self.ax2.set_xlabel('Freq (Hz)')
        self.ax2.set_ylabel('Magnitude')
        self.ax2.set_yscale('log')
        self.ax3.set_xlabel('shift')
        self.ax3.set_ylabel('Magnitude')
        self.ax2.set_ylim([1, 10000])
        self.ax2.set_xlim([0, 5000])

    def plot(self):
        total_duration = self.signal_length / self.sampling_rate
        n_section = math.ceil(total_duration / (self.frame_size / 1000))
        max_y = np.max(self.audio_signal_array)
        for section_num in range(0, n_section):
            section = self.audio_signal_array[
                      section_num * self.sample_per_section: min((section_num + 1) * self.sample_per_section,
                                                                 self.signal_length)]  # chop into section
            frequency_autocorr = autocorr_freq(section, self.sampling_rate)  # autocorr
            frequency_fft, signal_amplitude = generate_freq_spectrum(section, self.sampling_rate)  # fft
            peak_frequency_index_fft = np.argmax(
                signal_amplitude)  # get peak freq, return the index of the highest value

            if frequency_autocorr > 0 :
                self.ax1.text(section_num * self.sample_per_section, 0,
                              frequency_to_note(frequency_autocorr),
                              fontdict=self.font)  # plot note name
                #self.ax1.text(section_num * self.sample_per_section, 0.25 * max_y,
                              #frequency_to_note(frequency_fft[peak_frequency_index_fft]), fontdict = self.font)
                # self.ax1.text(section_num * self.sample_per_section, 0.25 * max_y, int(frequency_autocorr),
                # fontdict = self.font)  # freq
                self.ax1.axvline(x=min((section_num + 1) * self.sample_per_section, self.signal_length), color='r',
                                 linewidth=0.5,
                                 linestyle="-", zorder=10)  # lines for separating segments
                self.ax1.plot(self.audio_signal_array, zorder=0)

                self.start_indicator = self.ax1.axvline(
                    x=min(self.frame_slider.val * self.sample_per_section, self.signal_length), color='b',
                    linewidth=0.5,
                    linestyle="-", zorder=11)
                self.end_indicator = self.ax1.axvline(
                    x=min((self.frame_slider.val + 1) * self.sample_per_section, self.signal_length), color='b',
                    linewidth=0.5,
                    linestyle="-", zorder=11)

                # plot freq domain
                b1 = 0
                b2 = self.sample_per_section * self.frame_slider.val
                freq, magnitude = generate_freq_spectrum(self.audio_signal_array[b1:b2], self.sampling_rate)
                self.freq_line, = self.ax2.plot(freq, magnitude)  # add in peak points and text  #change to bar graph

    def exec_graph(self):
        self.frame_slider.on_changed(self.update)
        plt.show()

    def update(self, val):
        section = self.audio_signal_array[self.frame_slider.val * self.sample_per_section: min(
            (self.frame_slider.val + 1) * self.sample_per_section,
            self.signal_length)]  # chop into section
        # add try catch for shape mismatch
        frequency, signal_amplitude = generate_freq_spectrum(section, self.sampling_rate)  # fft

        autocorr_value = autocorr_freq(section)  # Replace signal_amplitude with section for time domain
        phase_zero_height = np.max(autocorr_value)
        peaks = find_peaks(autocorr_value, height=phase_zero_height / 2, distance=11)[
            0]  # find peaks return a tuple. but second item is empty
        dict = {}  # key is index
        for i in range(len(peaks)):
            peak_index = peaks[i]
            dict[autocorr_value[peak_index]] = peak_index
        sorted_value = sorted(dict.keys(), reverse=True)
        freq_from_corr = self.sampling_rate / abs(dict[sorted_value[0]] - dict[sorted_value[1]])
        self.ax3.clear()
        self.ax3.plot(autocorr_value)
        if phase_zero_height > 10000000:  # 10M
            self.ax3.text(0, 0, int(freq_from_corr),
                          fontdict=self.font)  # freq
            self.ax3.plot(peaks, autocorr_value[peaks], "x")

        self.ax1.lines.remove(self.start_indicator)
        self.ax1.lines.remove(self.end_indicator)
        self.start_indicator = self.ax1.axvline(
            x=min(self.frame_slider.val * self.sample_per_section, self.signal_length),
            color='b', linewidth=0.5,
            linestyle="-", zorder=11)
        self.end_indicator = self.ax1.axvline(
            x=min((self.frame_slider.val + 1) * self.sample_per_section, self.signal_length), color='b', linewidth=0.5,
            linestyle="-", zorder=11)
        self.freq_line.set_ydata(
            signal_amplitude)  # Replace signal_amplitude with autocorr_amplitude for autocorrelation plot
        self.fig.canvas.draw_idle()

    def to_do(self):
        # check why autocorrelate doesnt work with filtered signal
        pass


if __name__ == "__main__":
    sr, signall = path_to_numpy("test c4c5.wav")  # sr= 48k
    test = InteractivePlot(signall, sr, 200, 0)  # frame size in ms
    test.plot()
    test.exec_graph()
