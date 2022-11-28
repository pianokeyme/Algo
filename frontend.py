import math
import numpy as np
from test import generate_freq_spectrum, frequency_to_note, path_to_numpy, autocorr_freq


class Frontend:
    def __init__(self, threshold, sampling_rate):
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.prev_note_taken = ""
        self.prev_autocorr_note = ""
        self.curr_autocorr_note = ""
        self.curr_fft_note = ""
        self.prev_fft_note = ""
        self.autocorr_freq = 0
        self.fft_freq = 0

    def analysis(self, amplitudes_array):  # return 0 if no note detected
        frequency_autocorr = autocorr_freq(amplitudes_array, self.sampling_rate)  # autocorr
        frequency_fft, signal_amplitude = generate_freq_spectrum(amplitudes_array, self.sampling_rate)  # fft
        peak_frequency_index_fft = np.argmax(signal_amplitude)
        self.prev_autocorr_note = self.curr_autocorr_note
        self.prev_fft_note = self.curr_fft_note
        # autocorrelate
        if frequency_autocorr > 0:
            self.curr_autocorr_note = frequency_to_note(frequency_autocorr)
            self.autocorr_freq = frequency_autocorr
        else:
            self.curr_autocorr_note = ""

        # FFT
        if signal_amplitude[peak_frequency_index_fft] > self.threshold:
            self.curr_fft_note = frequency_to_note(frequency_fft[peak_frequency_index_fft])
            self.fft_freq = frequency_fft[peak_frequency_index_fft]
        else:
            self.curr_fft_note = ""

        frequency_taken, note_taken = self.chooseResult()
        self.prev_note_taken = note_taken

        if note_taken:
            o_note = ''.join([i for i in note_taken if i.isdigit()])
            o_octave = ''.join([i for i in note_taken if not i.isdigit()])
            return o_note, o_octave
        else:
            return 0

    def chooseResult(self):
        if self.curr_autocorr_note and self.curr_fft_note:  # Curr note detected from both
            if self.curr_autocorr_note == self.curr_fft_note:  # Curr notes are equivalent
                if self.prev_autocorr_note == self.prev_fft_note:  # Prev notes are equivalent so take that
                    return self.autocorr_freq, self.prev_autocorr_note
                else:  # Prev notes are not equivalent so take curr notes
                    return self.autocorr_freq, self.curr_autocorr_note
            else:  # Curr notes are different
                if self.curr_autocorr_note == self.prev_autocorr_note or self.curr_autocorr_note == self.prev_fft_note:  # Autocorrelate note same as at least half
                    return self.autocorr_freq, self.curr_autocorr_note
                elif self.curr_fft_note == self.prev_autocorr_note or self.curr_fft_note == self.prev_fft_note:  # FFT note same as at least half
                    return self.fft_freq, self.curr_fft_note
                else:
                    return 0, self.prev_note_taken
        else:  # signal dropped from one algo
            return 0, ""


if __name__ == "__main__":
    sr, signal = path_to_numpy("test c4c5.wav")  # sr= 48k
    test = Frontend(60, sr)
    signal_length = len(signal)
    frame_size = 200
    sample_per_section = int((frame_size / 1000) *sr)
    total_duration = signal_length / sr
    n_section = math.ceil(total_duration / (frame_size / 1000))
    for section_num in range(0, n_section):

        section = signal[section_num * sample_per_section: min((section_num + 1) * sample_per_section,
                                                               signal_length)]  # chop into section
        frequency_autocorr = autocorr_freq(section, sr)
        t = test.analysis(section)
        print(t)



