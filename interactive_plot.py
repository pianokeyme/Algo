import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.widgets import Slider
from test import bandpass_filter, generate_freq_spectrum, frequency_to_note, path_to_numpy
import time


note_string = ['C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B']


class InteractivePlot:

    def __init__(self, audio_signal_array, sampling_rate, frame_size, threshold ):
        self.audio_signal_array = audio_signal_array
        self.sampling_rate = sampling_rate
        self.frame_size = frame_size #in ms
        self.threshold = threshold
        self.signal_length = len(self.audio_signal_array)
        self.sample_per_section = int((self.frame_size / 1000) * self.sampling_rate)
        #function check and replot

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, figsize = (5,8))
        self.font = {'family': 'serif',
                     'color': 'darkred',
                     'weight': 'normal',
                     'size': 5,
                     }
        axfreq = self.fig.add_axes([0.25, 0.1, 0.65, 0.03])
        self.fig.subplots_adjust(left=0.25, bottom=0.25)
        total_duration = self.signal_length / self.sampling_rate
        n_section= math.ceil(total_duration / (self.frame_size / 1000))
        self.frame_slider = Slider(    ######can only increment by 2 for some reason. even when step is step at 1
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
        self.ax2.set_ylim([0,2000])

    def plot(self):
        total_duration = self.signal_length / self.sampling_rate
        n_section= math.ceil(total_duration / (self.frame_size / 1000))
        max_y = np.max(self.audio_signal_array)
        for section_num in range(0, n_section):
            section = self.audio_signal_array[section_num * self.sample_per_section: min((section_num + 1) * self.sample_per_section,
                                                                         self.signal_length)]  # chop into section
            frequency, signal_amplitude = generate_freq_spectrum(section, self.sampling_rate)  # fft
            peak_frequency_index = np.argmax(
                signal_amplitude)  # get peak freq, return the index of the highest value
            if signal_amplitude[peak_frequency_index] > self.threshold:
                self.ax1.text(section_num * self.sample_per_section, 0, frequency_to_note(frequency[peak_frequency_index]),
                         fontdict=self.font)  # plot note name
                self.ax1.text(section_num * self.sample_per_section, 0.25 * max_y, frequency[peak_frequency_index],
                         fontdict=self.font)  # freq
            self.ax1.axvline(x=min((section_num + 1) * self.sample_per_section, self.signal_length), color='r', linewidth=0.5,
                        linestyle="-", zorder=10)  # lines for separating segments
            self.ax1.text(section_num * self.sample_per_section, 0.5 * max_y, round(signal_amplitude[peak_frequency_index]),
                     fontdict=self.font)  # plot the freq magnitude
        self.ax1.plot(self.audio_signal_array, zorder=0)

        self.start_indicator = self.ax1.axvline(x=min(self.frame_slider.val * self.sample_per_section, self.signal_length), color='b', linewidth=0.5,
                        linestyle="-", zorder=11)
        self.end_indicator = self.ax1.axvline(x=min((self.frame_slider.val + 1) * self.sample_per_section, self.signal_length), color='b', linewidth=0.5,
                        linestyle="-", zorder=11)

        #plot freq domain
        b1 = 0
        b2 = self.sample_per_section* self.frame_slider.val
        filtered_signal = bandpass_filter(self.audio_signal_array, self.sampling_rate)
        freq, magnitude = generate_freq_spectrum(filtered_signal[b1:b2], self.sampling_rate)
        self.freq_line, = self.ax2.plot(freq, magnitude) #add in peak points and text  #change to bar graph
        print(str(len(magnitude))+ " initial")

    def exec_graph(self):
        self.frame_slider.on_changed(self.update)
        plt.show()

    def update(self, val):
        section = self.audio_signal_array[self.frame_slider.val * self.sample_per_section: min((self.frame_slider.val + 1) * self.sample_per_section,
                                                                        self.signal_length)]  # chop into section
        #add try catch for shape mismatch
        print(str(len(section)) + "update section len")
        frequency, signal_amplitude = generate_freq_spectrum(section, self.sampling_rate)  # fft
        print(str(len(signal_amplitude)) + "amp len update")
        self.ax1.lines.remove(self.start_indicator)
        self.ax1.lines.remove(self.end_indicator)
        self.start_indicator = self.ax1.axvline(x=min(self.frame_slider.val * self.sample_per_section, self.signal_length),
                                                color='b', linewidth=0.5,
                                                linestyle="-", zorder=11)
        self.end_indicator = self.ax1.axvline(
            x=min((self.frame_slider.val + 1) * self.sample_per_section, self.signal_length), color='b', linewidth=0.5,
            linestyle="-", zorder=11)
        self.freq_line.set_ydata(signal_amplitude)
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    sr, signall = path_to_numpy("test c4c5.wav") #sr= 48k
    print(sr)
    test = InteractivePlot(signall, sr, 50, 50)
    test.plot()
    test.exec_graph()


