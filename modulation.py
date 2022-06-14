import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from bandpass import temporal_bandpass_filter
from scipy.spatial.distance import cdist
from hamming_code import Hamming


class DataError(Exception):
    pass


class Modulator:
    """Parent class for various modulation schemes containing the most important parameters."""
    def __init__(self, data_rate, carrier_frequency, carrier_amplitude,
                 sampling_frequency, modulation_index=0.5, M=2, coding=False, j=3):
        if coding:
            self.coding = Hamming(j)
        else:
            self.coding = None
        self.j = j
        self.data_rate = data_rate
        self.carrier_frequency = carrier_frequency
        self.carrier_amplitude = carrier_amplitude
        self.fs = sampling_frequency
        self.modulation_index = modulation_index
        if (M == 1) or (M > 64) or (np.mod(np.log2(M), 1.0) != 0.0):
            raise ValueError("M must be a positive power of 2 (2, 4, 8, 16, 32, 64)")
        self.M = M

        self.bits_per_symbol = int(np.log2(self.M))                # bits per data symbol
        self.symbol_rate = data_rate / self.bits_per_symbol        # derive the symbol rate
        self.samples_per_symbol = int(self.fs / self.symbol_rate)  # DAC/ADC bits per symbol
        self.signal_length = None                                  # signal length in samples
        self.time = None                                           # time axis of the signal
        self.data_bits = None                                      # bits to be transmitted
        self.data_symbols = None                                   # symbols to be transmitted
        self.signal = None                                         # signal to be transmitted
        self.mapping_table = None                 # table to map symbols to constellations
        self.psd = None                           # power spectral density of signal
        self.detected_symbols = []         # list to save detected constellation points
        self.received_symbols = []

    def set_data(self, data):
        """Validates the given data and stores it as a class attribute."""
        if type(data) != list and not isinstance(data, np.ndarray):
            raise DataError("Invalid format. Please supply data bits as a Python list.")
        elif any(bit not in [0, 1] for bit in data):
            raise DataError("Invalid bits. Please use only the symbols 0 and 1.")

        # set data and divide the bit sequence into symbols of the correct length
        self.data_bits_uncoded = data
        if self.coding:
            self.data_bits = self.coding.encode(data)
        else:
            self.data_bits = data
        self.generate_symbols(int(np.log2(self.M)))
        self.signal_length = int(len(self.data_symbols) * self.samples_per_symbol)
        self.time = np.linspace(1/self.fs, self.signal_length/self.fs, self.signal_length)

    def decode_data(self, data_encoded):
        if self.coding:
            return self.coding.decode(data_encoded)
        else:
            return data_encoded

    def generate_symbols(self, bits_per_symbol):
        """Splits the input data sequence into symbols of the correct length."""
        if len(self.data_bits) % bits_per_symbol != 0:  # check if length divisible by M
            self.data_bits = np.append(self.data_bits,
                                       np.zeros(len(self.data_bits) % bits_per_symbol))
        self.data_symbols = np.reshape(self.data_bits,
                            (int(len(self.data_bits)/bits_per_symbol), bits_per_symbol))

    def bit_error_rate(self, data, additive_noise):
        """Computes the bit error rate (BER) for the given data bits and additive noise."""
        signal = self.modulate(data)                         # modulate data
        signal_with_noise = np.add(signal, additive_noise)      # add noise
        decoded_data = self.demodulate(signal_with_noise)    # demodulate signal
        num_errors = np.sum(np.abs(np.subtract(np.array(data), np.array(decoded_data))))
        ber = num_errors/len(data)                              # bit error rate

        return signal_with_noise, ber

    def iq_mapping(self):
        # received vector and reference in cartesian form
        constellations = np.array([v for k, v in self.mapping_table.items()])
        XA = np.column_stack((np.real(self.received_symbols), np.imag(self.received_symbols)))
        XB = np.column_stack((np.real(constellations), np.imag(constellations)))
        d = cdist(XA, XB, metric='euclidean')  # compute pair-wise Euclidean distances
        self.detected_symbols = constellations[np.argmin(d, axis=1)]  # indices corresponding minimum Euclid. dist.
        return self.detected_symbols

    def spectrum(self):
        """Returns the frequency spectrum of the modulated signal."""
        signal_spectrum = np.abs(np.fft.fft(self.signal))       # compute spectrum with FFT
        signal_spectrum /= max(signal_spectrum)                 # normalize spectrum
        signal_spectrum_db = 10*np.log10(signal_spectrum)       # convert to dB scale
        frequency_axis = np.linspace(0, self.fs/2, int(len(self.signal)/2))  # frequency axis
        return signal_spectrum_db[0:int(len(signal_spectrum)/2)], frequency_axis

    def signal_power(self):
        """Returns the total power of the modulated signal."""
        signal_power = np.sum(np.abs(self.signal)**2)/(len(self.signal))
        return signal_power

    def generate_mapping_table(self, states):
        """Generates table to map symbols to constellation points or
           frequencies given by 'states'."""
        if self.M == 2:
            self.mapping_table = {
                (0,): states[0],
                (1,): states[1]
            }
        elif self.M == 4:
            self.mapping_table = {
                (0, 0): states[0],
                (0, 1): states[1],
                (1, 0): states[2],
                (1, 1): states[3]
            }
        elif self.M == 8:
            self.mapping_table = {
                (0, 0, 0): states[0],
                (0, 0, 1): states[1],
                (0, 1, 0): states[2],
                (0, 1, 1): states[3],
                (1, 0, 0): states[4],
                (1, 0, 1): states[5],
                (1, 1, 0): states[6],
                (1, 1, 1): states[7]
            }
        elif self.M == 16:
            self.mapping_table = {
                (0, 0, 0, 0): states[0],
                (0, 0, 0, 1): states[1],
                (0, 0, 1, 0): states[2],
                (0, 0, 1, 1): states[3],
                (0, 1, 0, 0): states[4],
                (0, 1, 0, 1): states[5],
                (0, 1, 1, 0): states[6],
                (0, 1, 1, 1): states[7],
                (1, 0, 0, 0): states[8],
                (1, 0, 0, 1): states[9],
                (1, 0, 1, 0): states[10],
                (1, 0, 1, 1): states[11],
                (1, 1, 0, 0): states[12],
                (1, 1, 0, 1): states[13],
                (1, 1, 1, 0): states[14],
                (1, 1, 1, 1): states[15]
            }
        elif self.M == 32:
            self.mapping_table = {
                (0, 0, 0, 0, 0): states[0],
                (0, 0, 0, 0, 1): states[1],
                (0, 0, 0, 1, 0): states[2],
                (0, 0, 0, 1, 1): states[3],
                (0, 0, 1, 0, 0): states[4],
                (0, 0, 1, 0, 1): states[5],
                (0, 0, 1, 1, 0): states[6],
                (0, 0, 1, 1, 1): states[7],
                (0, 1, 0, 0, 0): states[8],
                (0, 1, 0, 0, 1): states[9],
                (0, 1, 0, 1, 0): states[10],
                (0, 1, 0, 1, 1): states[11],
                (0, 1, 1, 0, 0): states[12],
                (0, 1, 1, 0, 1): states[13],
                (0, 1, 1, 1, 0): states[14],
                (0, 1, 1, 1, 1): states[15],
                (1, 0, 0, 0, 0): states[16],
                (1, 0, 0, 0, 1): states[17],
                (1, 0, 0, 1, 0): states[18],
                (1, 0, 0, 1, 1): states[19],
                (1, 0, 1, 0, 0): states[20],
                (1, 0, 1, 0, 1): states[21],
                (1, 0, 1, 1, 0): states[22],
                (1, 0, 1, 1, 1): states[23],
                (1, 1, 0, 0, 0): states[24],
                (1, 1, 0, 0, 1): states[25],
                (1, 1, 0, 1, 0): states[26],
                (1, 1, 0, 1, 1): states[27],
                (1, 1, 1, 0, 0): states[28],
                (1, 1, 1, 0, 1): states[29],
                (1, 1, 1, 1, 0): states[30],
                (1, 1, 1, 1, 1): states[31]

            }
        elif self.M == 64:
            self.mapping_table = {
                (0, 0, 0, 0, 0, 0): states[0],
                (0, 0, 0, 0, 0, 1): states[1],
                (0, 0, 0, 0, 1, 0): states[2],
                (0, 0, 0, 0, 1, 1): states[3],
                (0, 0, 0, 1, 0, 0): states[4],
                (0, 0, 0, 1, 0, 1): states[5],
                (0, 0, 0, 1, 1, 0): states[6],
                (0, 0, 0, 1, 1, 1): states[7],
                (0, 0, 1, 0, 0, 0): states[8],
                (0, 0, 1, 0, 0, 1): states[9],
                (0, 0, 1, 0, 1, 0): states[10],
                (0, 0, 1, 0, 1, 1): states[11],
                (0, 0, 1, 1, 0, 0): states[12],
                (0, 0, 1, 1, 0, 1): states[13],
                (0, 0, 1, 1, 1, 0): states[14],
                (0, 0, 1, 1, 1, 1): states[15],
                (0, 1, 0, 0, 0, 0): states[16],
                (0, 1, 0, 0, 0, 1): states[17],
                (0, 1, 0, 0, 1, 0): states[18],
                (0, 1, 0, 0, 1, 1): states[19],
                (0, 1, 0, 1, 0, 0): states[20],
                (0, 1, 0, 1, 0, 1): states[21],
                (0, 1, 0, 1, 1, 0): states[22],
                (0, 1, 0, 1, 1, 1): states[23],
                (0, 1, 1, 0, 0, 0): states[24],
                (0, 1, 1, 0, 0, 1): states[25],
                (0, 1, 1, 0, 1, 0): states[26],
                (0, 1, 1, 0, 1, 1): states[27],
                (0, 1, 1, 1, 0, 0): states[28],
                (0, 1, 1, 1, 0, 1): states[29],
                (0, 1, 1, 1, 1, 0): states[30],
                (0, 1, 1, 1, 1, 1): states[31],
                (1, 0, 0, 0, 0, 0): states[32],
                (1, 0, 0, 0, 0, 1): states[33],
                (1, 0, 0, 0, 1, 0): states[34],
                (1, 0, 0, 0, 1, 1): states[35],
                (1, 0, 0, 1, 0, 0): states[36],
                (1, 0, 0, 1, 0, 1): states[37],
                (1, 0, 0, 1, 1, 0): states[38],
                (1, 0, 0, 1, 1, 1): states[39],
                (1, 0, 1, 0, 0, 0): states[40],
                (1, 0, 1, 0, 0, 1): states[41],
                (1, 0, 1, 0, 1, 0): states[42],
                (1, 0, 1, 0, 1, 1): states[43],
                (1, 0, 1, 1, 0, 0): states[44],
                (1, 0, 1, 1, 0, 1): states[45],
                (1, 0, 1, 1, 1, 0): states[46],
                (1, 0, 1, 1, 1, 1): states[47],
                (1, 1, 0, 0, 0, 0): states[48],
                (1, 1, 0, 0, 0, 1): states[49],
                (1, 1, 0, 0, 1, 0): states[50],
                (1, 1, 0, 0, 1, 1): states[51],
                (1, 1, 0, 1, 0, 0): states[52],
                (1, 1, 0, 1, 0, 1): states[53],
                (1, 1, 0, 1, 1, 0): states[54],
                (1, 1, 0, 1, 1, 1): states[55],
                (1, 1, 1, 0, 0, 0): states[56],
                (1, 1, 1, 0, 0, 1): states[57],
                (1, 1, 1, 0, 1, 0): states[58],
                (1, 1, 1, 0, 1, 1): states[59],
                (1, 1, 1, 1, 0, 0): states[60],
                (1, 1, 1, 1, 0, 1): states[61],
                (1, 1, 1, 1, 1, 0): states[62],
                (1, 1, 1, 1, 1, 1): states[63]
            }
        self.demapping_table = {v: k for k, v in self.mapping_table.items()}

    def modulate(self, data):
        pass

    def demodulate(self, signal):
        pass


# The modulator implementations below are not used anymore.
# New modulator implementations are located in the 'modulators' directory

class AmplitudeShiftKeying(Modulator):
    def __init__(self, data_rate, carrier_frequency, carrier_amplitude, sampling_frequency,
                 modulation_index, M):
        super().__init__(data_rate, carrier_frequency, carrier_amplitude, sampling_frequency,
                         modulation_index, M)
        self.theoretical_psd()  # compute the theoretical psd of the signal

    def modulate(self, data):
        """Converts the given data bits to an ASK modulated signal."""
        self.set_data(data)

        # set amplitude level for each data symbol and generate mapping table
        levels = np.linspace(-1, 1, self.M)
        states = self.carrier_amplitude*(1 + self.modulation_index*levels)
        self.generate_mapping_table(states)

        # create amplitude vector given the symbols to be transmitted
        amplitude = [self.mapping_table[tuple(symbol)] for symbol in self.data_symbols]
        # extend the amplitude vector to cover all signal samples
        amplitude = np.repeat(amplitude, self.samples_per_symbol)

        # construct signal
        self.signal = amplitude * np.sin(2 * np.pi * self.carrier_frequency * self.time)
        return self.signal

    def demodulate(self, signal):
        """Demodulates the given ASK signal to retrieve the data bits"""
        # filter out noise and determine the signal envelope using Hilbert transform
        signal = temporal_bandpass_filter(signal, self.fs, 0.8*self.carrier_frequency,
                                          1.2*self.carrier_frequency)
        hilbert_signal = hilbert(signal)
        signal_envelope = np.abs(hilbert_signal)

        # find number of bits (assuming the entire signal contains useful bits)
        num_bits = (int(len(signal)-self.samples_per_symbol/2)//self.samples_per_symbol) + 1
        # find data bits by sampling signal envelope in the middle of the signaling interval
        data = []
        for i in range(num_bits):
            # sample signal envelope
            measured_envelope = signal_envelope[int(self.samples_per_symbol / 2) +
                                                i * self.samples_per_symbol]
            bits_estimate = None
            min_distance = np.Inf
            # find amplitude constellation closest to measured value
            for bit_sequence, value in self.mapping_table.items():
                d = abs(value - measured_envelope)
                if d <= min_distance:
                    min_distance = d
                    bits_estimate = list(bit_sequence)
                self.detected_symbols.append(measured_envelope)
            data.extend(bits_estimate)
        return self.decode_data(data)

    def theoretical_psd(self):
        """Computes the theoretical spectrum of the signal."""
        f_axis = np.linspace(0, self.fs, 10000) - self.carrier_frequency
        delta = np.zeros(len(f_axis))
        delta[int(10000 * self.carrier_frequency / self.fs)] = 1
        self.psd = (self.carrier_amplitude ** 2) * (delta + (np.sin(
            np.pi * f_axis / self.symbol_rate) / (np.pi * f_axis / self.symbol_rate)) ** 2 /
                                                    self.symbol_rate) / 2


class FrequencyShiftKeying(Modulator):
    def __init__(self, data_rate, carrier_frequency, carrier_amplitude, sampling_frequency,
                 modulation_index, M):
        super().__init__(data_rate, carrier_frequency, carrier_amplitude, sampling_frequency,
                         modulation_index, M)
        self.theoretical_psd()

    def modulate(self, data):
        """Converts the given data bits to an FSK modulated signal."""
        self.set_data(data)

        # set frequency level for each data symbol and generate mapping table
        levels = np.concatenate((np.linspace(-self.M + 0.5, -0.5, int(self.M / 2)),
                                 np.linspace(0.5, self.M - 0.5, int(self.M / 2))))
        if self.M == 2:
            levels = np.array([-0.5, 0.5])
        states = self.carrier_frequency + self.modulation_index * self.symbol_rate * levels
        self.generate_mapping_table(states)

        # create frequency vector given the symbols to be transmitted
        frequency = [self.mapping_table[tuple(symbol)] for symbol in self.data_symbols]
        # extend the frequency vector to cover all signal samples
        frequency = np.repeat(frequency, self.samples_per_symbol)

        # construct the signal
        self.signal = self.carrier_amplitude * np.sin(2 * np.pi * frequency * self.time)
        return self.signal

    def demodulate(self, signal):
        """Demodulates the given FSK signal to retrieve the data bits"""
        # get frequency levels
        levels = np.concatenate((np.linspace(-self.M + 0.5, -0.5, int(self.M / 2)),
                                 np.linspace(0.5, self.M - 0.5, int(self.M / 2))))
        if self.M == 2:
            levels = np.array([-0.5, 0.5])
        frequencies = self.carrier_frequency+self.modulation_index*self.symbol_rate*levels
        # find spacing between adjacent frequency levels
        spacing = abs(frequencies[1] - frequencies[0])

        # create arrays to store bandpass-filtered copies of the signal and their envelopes
        filtered_signals = np.ndarray((self.M, len(signal)))
        envelopes = np.ndarray((self.M, len(signal)))

        # apply bandpass filters to discriminate between frequencies in the signal
        for i, freq in enumerate(frequencies):
            filtered_signals[i] = temporal_bandpass_filter(signal, self.fs, freq - 0.5 *
                                                           spacing, freq + 0.5 * spacing)
            envelopes[i] = np.abs(hilbert(filtered_signals[i]))

        # find number of bits (assuming the entire signal contains useful bits)
        num_bits = (int(len(signal)-self.samples_per_symbol/2)//self.samples_per_symbol) + 1

        # derive de-mapping table to map frequencies to back to symbols
        demapping_table = {v: k for k, v in self.mapping_table.items()}
        # find data bits by sampling the frequency in the middle of the signaling interval
        data = []
        for i in range(num_bits):
            amplitudes = envelopes[:, int(self.samples_per_symbol / 2) + i *
                                   self.samples_per_symbol]
            freq = frequencies[np.argmax(amplitudes, axis=0)]

            data.extend(list(demapping_table[int(freq)]))
        return self.decode_data(data)

    def theoretical_psd(self):
        """Computes the theoretical spectrum of the signal."""
        f_axis = np.linspace(0, self.fs, 10000)

        self.psd = 2 * self.carrier_amplitude ** 2 / (np.pi ** 2 * self.data_rate) * (
                    np.cos(np.pi * (f_axis - self.carrier_frequency) / self.data_rate) / (
                        4 * (f_axis-self.carrier_frequency)**2/self.data_rate**2-1))**2
        hz_per_psd_sample = int(self.fs / len(f_axis))
        self.psd[int(self.carrier_frequency / hz_per_psd_sample + self.data_rate / (
                    2 * hz_per_psd_sample))] += self.carrier_amplitude ** 2 / 8
        self.psd[int(self.carrier_frequency / hz_per_psd_sample - self.data_rate / (
                    2 * hz_per_psd_sample))] += self.carrier_amplitude ** 2 / 8


class PhaseShiftKeying(Modulator):
    def __init__(self, data_rate, carrier_frequency, carrier_amplitude, sampling_frequency,
                 modulation_index, M):
        super().__init__(data_rate, carrier_frequency, carrier_amplitude, sampling_frequency,
                         modulation_index, M)
        self.theoretical_psd()

    def modulate(self, data):
        """Converts the given data bits to a PSK modulated signal."""
        self.set_data(data)

        # set phase levels corresponding to each data symbol and generate mapping table
        m = np.arange(0, self.M)  # all information symbols m={0,1,...,M-1}
        I = 1 / np.sqrt(2) * np.cos(m / self.M * 2 * np.pi)
        Q = 1 / np.sqrt(2) * np.sin(m / self.M * 2 * np.pi)
        states = np.angle(I - 1j*Q)
        self.generate_mapping_table(states)

        # create phase vector given the symbols to be transmitted
        phase = [self.mapping_table[tuple(symbol)] for symbol in self.data_symbols]
        # extend the phase vector to cover all signal samples
        phase = np.repeat(phase, self.samples_per_symbol)
        if all(np.imag(phase) < 10**(-9)):
            phase = np.real(phase)
        # construct the signal
        print(phase)
        self.signal = self.carrier_amplitude * np.sin(
            2 * np.pi * self.carrier_frequency * self.time + phase)
        return self.signal

    def demodulate(self, signal):
        """Demodulates the given PSK signal to retrieve the data bits"""
        # filter out noise and determine the instantaneous phase using the Hilbert transform
        signal = temporal_bandpass_filter(signal, self.fs, 0.8*self.carrier_frequency,
                                          1.2*self.carrier_frequency)
        hilbert_signal = hilbert(signal)
        instantaneous_phase = np.unwrap(np.angle(hilbert_signal))
        # correct for the frequency of the signal (time-varying phase)
        corrected_phase = (instantaneous_phase - 2*np.pi*self.carrier_frequency*self.time) \
                           % (2*np.pi)
        dist_between_phases = 2*np.pi/self.M

        # find number of bits (assuming the entire signal contains useful bits)
        num_bits = (int(len(signal)-self.samples_per_symbol/2)//self.samples_per_symbol) + 1

        # fill in the data bits by sampling the phase in the middle of the bit
        data = []
        for i in range(num_bits):
            measured_phase = corrected_phase[int(self.samples_per_symbol / 2) + i *
                                             self.samples_per_symbol]
            if 2*np.pi-dist_between_phases/2 < measured_phase < 2*np.pi:
                measured_phase = 0
            bits_estimate = None
            min_distance = np.Inf
            # find phase constellation closest to measured value
            for bit_sequence, value in self.mapping_table.items():
                d = abs(value - measured_phase) % 2*np.pi
                if d <= min_distance:
                    min_distance = d
                    bits_estimate = list(bit_sequence)
                elif abs(d-2*np.pi) <= min_distance:
                    min_distance = abs(d-2*np.pi)
                    bits_estimate = list(bit_sequence)
                self.detected_symbols.append(measured_phase)
            data.extend(bits_estimate)
        return self.decode_data(data)

    def theoretical_psd(self):
        """Computes the theoretical spectrum of the signal."""
        f_axis = np.linspace(0, self.fs, 10000) - self.carrier_frequency
        self.psd = (self.carrier_amplitude**2)*(np.sin(np.pi*f_axis/self.symbol_rate)/(
                    np.pi * f_axis / self.symbol_rate)) ** 2  # /self.symbol_rate


class QAM(Modulator):
    def __init__(self, data_rate, carrier_frequency, carrier_amplitude, sampling_frequency,
                 modulation_index, M=4):
        super().__init__(data_rate, carrier_frequency, carrier_amplitude, sampling_frequency,
                         modulation_index, M)
        # check if the given M is a valid number of constellations for square M-QAM
        if (M == 1) or (M > 64) or (np.mod(np.log2(M), 2) != 0):
            raise ValueError("Only square M-QAM up to M=64 supported. M must be an"
                             " even power of 2 (4, 16, or 64)")
        else:
            self.m = M      # number of states

        # define complex constellation points and generate mapping table
        if self.m == 4:
            states = [-1+1j, 1+1j, -1-1j, 1-1j]
        elif self.m == 16:
            states = [-3-3j, -3-1j, -3+3j, -3+1j, -1-3j, -1-1j, -1+3j,
                      -1+1j, 3-3j, 3-1j, 3+3j, 3+1j, 1-3j, 1-1j, 1+3j, 1+1j]
        else:   # m = 64
            states = [7+7j, 5+7j, 1+7j, 3+7j, -7+7j, -5+7j, -1+7j, -3+7j, 7+5j, 5+5j, 1+5j,
                      3+5j, -7+5j, -5+5j, -1+5j, -3+5j, 7+1j, 5+1j, 1+1j, 3+1j, -7+1j, -5+1j,
                      -1+1j, -3+1j, 7+3j, 5+3j, 1+3j, 3+3j, -7+3j, -5+3j, -1+3j, -3+3j, 7-7j,
                      5-7j, 1-7j, 3-7j, -7-7j, -5-7j, -1-7j, -3-7j, 7-5j, 5-5j, 1-5j, 3-5j,
                      -7-5j, -5-5j, -1-5j, -3-5j, 7-1j, 5-1j, 1-1j, 3-1j, -7-1j, -5-1j,
                      -1-1j, -3-1j, 7-3j, 5-3j, 1-3j, 3-3j, -7-3j, -5-3j, -1-3j, -3-3j]
        self.generate_mapping_table(states)
        self.generate_psd()

    def generate_psd(self):
        """Computes the theoretical spectrum of the signal."""
        # compute variance C between levels
        C = np.mean(np.abs(np.array([i for i in self.mapping_table.values()]))**2)
        K = C*np.log2(self.m)/self.data_rate
        f_axis = np.linspace(0, self.fs, 10000) - self.carrier_frequency
        self.psd = (np.sin(np.pi*f_axis*np.log2(self.m)/self.data_rate)/
                    (np.pi*f_axis*np.log2(self.m)/self.data_rate))**2  # *K

    def modulate(self, data):
        """Converts the given data bits to a QAM modulated signal."""
        self.set_data(data)

        # define amplitude and phase vectors given the data symbols
        constellations = np.array([self.mapping_table[tuple(symbol)] for symbol in
                                   self.data_symbols])
        amplitude = self.carrier_amplitude*abs(constellations)
        phase = np.angle(constellations)
        # extend the amplitude/phase vectors to cover all signal samples
        amplitude = np.repeat(amplitude, self.samples_per_symbol)
        phase = np.repeat(phase, self.samples_per_symbol)

        # create signal
        self.signal = amplitude*np.cos(2 * np.pi * self.carrier_frequency*self.time + phase)
        return self.signal

    def demodulate(self, signal):
        """Demodulates the given QAM signal to retrieve the data bits."""
        # filter out noise and determine complex envelope using the hilbert transform
        signal = temporal_bandpass_filter(signal, self.fs, 0.9*self.carrier_frequency,
                                          1.1*self.carrier_frequency)
        hilbert_signal = hilbert(signal)
        # extract phase and envelope
        phase = np.angle(hilbert_signal)
        envelope = abs(hilbert_signal)
        # correct the phase and envelope for the phase deviation due to the frequency
        corrected_phase = (phase - 2*np.pi*self.carrier_frequency*self.time) % (2*np.pi)
        corrected_hilbert = envelope*np.exp(1j*corrected_phase)
        # find number of bits (assuming the entire signal contains useful bits)
        num_bits = (int(len(signal)-self.samples_per_symbol/2)//self.samples_per_symbol) + 1
        # find data bits by sampling complex envelope in the middle of the signaling interval
        data = []
        for i in range(num_bits):
            measured_constellation = corrected_hilbert[int(self.samples_per_symbol / 2) +
                                                       i * self.samples_per_symbol]
            bits_estimate = None
            min_distance = np.Inf
            # find the point in the I-Q plot closest to the measured value
            for bit_sequence, value in self.mapping_table.items():
                d = abs(value - measured_constellation)
                if d <= min_distance:
                    min_distance = d
                    bits_estimate = list(bit_sequence)
                self.detected_symbols.append(measured_constellation)
            data.extend(bits_estimate)
        return self.decode_data(data)

    def iq_plot(self):
        """Generates IQ plot (constellation diagram) showing the transmitted and
           detected constellation points."""
        p = plt.figure(figsize=(4, 4))
        for k, v in self.mapping_table.items():
            plt.plot(np.real(v), np.imag(v), markersize=3, marker='o', markeredgecolor='r',
                     markerfacecolor='r')
        for v in self.detected_symbols:
            plt.plot(np.real(v), np.imag(v), markersize=1, marker='o', markeredgecolor='k',
                     markerfacecolor='k')
        plt.vlines(0, ymin=-9, ymax=9, color='k')
        dist_between_dots = 2
        boundaries = [a*dist_between_dots for a in range(-int(np.sqrt(self.m)-1)//2,
                                                         int(np.sqrt(self.m)-1)//2+1)]
        plt.vlines(boundaries, ymin=-9, ymax=9, color='k', linestyles='--', linewidth=1)
        plt.hlines(boundaries, xmin=-9, xmax=9, color='k', linestyles='--', linewidth=1)
        plt.axhline(y=0, color='k')
        lim = dist_between_dots*np.sqrt(self.m)/2
        plt.ylim([-lim, lim])
        plt.xlim([-lim, lim])
        plt.xlabel("In-phase Amplitude")
        plt.ylabel("Quadrature Amplitude")
        plt.title(f"IQ plot {self.m}-QAM")
        #plt.show()
        return p


class DifferentialPhaseShiftKeying(Modulator):
    def __init__(self, data_rate, carrier_frequency, carrier_amplitude, sampling_frequency,
                 modulation_index, M):
        super().__init__(data_rate, carrier_frequency, carrier_amplitude, sampling_frequency,
                         modulation_index, M)

    def modulate(self, data):
        """Converts the given data bits to a DPSK modulated signal."""
        self.set_data(data)
        phase = []
        encoded_bit = 0
        for bit in self.data_bits:
            previous_encoded_bit = encoded_bit
            # apply differential encoding using the modulo 2 operation
            encoded_bit = np.mod(bit + previous_encoded_bit, 2)
            phase.append(encoded_bit*np.pi)

        phase = np.repeat(np.array(phase), self.samples_per_symbol)

        self.signal = self.carrier_amplitude * np.cos(
            2 * np.pi * self.carrier_frequency * self.time + phase)  # construct signal
        return self.signal

    def demodulate(self, signal):
        """Demodulates the given DPSK signal to retrieve the data bits"""
        # filter out noise and determine the instantaneous phase using the Hilbert transform
        signal = temporal_bandpass_filter(signal, self.fs, 0.9 * self.carrier_frequency,
                                          1.1 * self.carrier_frequency)
        hilbert_signal = hilbert(signal)
        instantaneous_phase = np.unwrap(np.angle(hilbert_signal))
        # correct the phase for the deviation caused by the signal frequency
        corrected_phase = (instantaneous_phase-2*np.pi*self.carrier_frequency*self.time) \
                           % (2 * np.pi)

        # find number of bits (assuming the entire signal contains useful bits)
        num_bits = (int(len(signal)-self.samples_per_symbol/2)//self.samples_per_symbol) + 1
        # fill in the data bits by sampling the phase in the middle of the signaling interval
        data = []
        prev_phase = 0
        not_prev_phase = np.pi
        for i in range(num_bits):
            measured_phase = corrected_phase[int(self.samples_per_symbol / 2) +
                                             i * self.samples_per_symbol]
            if 2 * np.pi > measured_phase > 1.5 * np.pi:
                measured_phase = measured_phase - 2 * np.pi

            if abs(measured_phase-prev_phase) < abs(measured_phase-not_prev_phase) and \
               abs(measured_phase-prev_phase) < abs(measured_phase-not_prev_phase-2*np.pi):
                data.append(0)
                prev_phase = prev_phase
                not_prev_phase = not_prev_phase
            else:
                data.append(1)
                if prev_phase == 0:
                    prev_phase = np.pi
                    not_prev_phase = 0
                else:
                    prev_phase = 0
                    not_prev_phase = np.pi
        return self.decode_data(data)


class Pi4QPhaseShiftKeying(Modulator):
    def __init__(self, data_rate, carrier_frequency, carrier_amplitude, sampling_frequency, modulation_index, M):
        super().__init__(data_rate, carrier_frequency, carrier_amplitude, sampling_frequency, modulation_index, M)
        self.bits_per_symbol = 2
        self.detected_values = []
        self.samples_per_symbol = int(2 * self.fs / self.data_rate)

    def modulate(self, data):
        """Converts the given data bits to a pi/4 DQPSK modulated signal."""
        self.set_data(data)
        symbols = np.reshape(data, (int(len(data)/2), 2))
        phase = []
        prev_phase = np.pi / 4
        for symbol in tuple(map(tuple, symbols)):
            if symbol == (0, 0):    # -135 degree phase shift
                prev_phase = prev_phase - (135 * 2*np.pi/360)
                phase.append(prev_phase)
            elif symbol == (0, 1):  # 135 degree phase shift
                prev_phase = prev_phase + (135 * 2*np.pi / 360)
                phase.append(prev_phase)
            elif symbol == (1, 0):  # -45 degree phase shift
                prev_phase = prev_phase - (45 * 2*np.pi / 360)
                phase.append(prev_phase)
            else:                   # 45 degree phase shift
                prev_phase = prev_phase + (45 * 2*np.pi / 360)
                phase.append(prev_phase)

        phase = np.repeat(phase, 2*self.samples_per_symbol)

        self.signal = self.carrier_amplitude * np.cos(
            2 * np.pi * self.carrier_frequency * self.time + phase)  # construct signal
        return self.signal

    def demodulate(self, signal):
        # Different phase shifts possible
        phase_shift_zz = -135*2*np.pi/360
        phase_shift_zo = 135*2*np.pi/360
        phase_shift_oz = -45*2*np.pi/360
        phase_shift_oo = 45*2*np.pi/360
        # Determine the instantaneous phase using the Hilbert transform
        signal = temporal_bandpass_filter(signal, self.fs, 0.8 * self.carrier_frequency, 1.2 * self.carrier_frequency)
        hilbert_signal = hilbert(signal)
        instantaneous_phase = np.unwrap(np.angle(hilbert_signal))
        corrected_phase = (instantaneous_phase - 2 * np.pi * self.carrier_frequency * self.time) % (2 * np.pi)
        # Find the number of bits using the length of the signal (assuming the entire signal contains useful bits)
        num_bits = (int(len(signal) - self.samples_per_symbol / 2) // self.samples_per_symbol) + 1
        # Fill in the data bits by checking the phase level in the middle of the bit
        data = []
        prev_phase = np.pi/4
        #not_prev_phase = np.pi
        for i in range(0, int(num_bits), 2):
            measured_phase = corrected_phase[int(0.9*self.samples_per_symbol) + i * self.samples_per_symbol]
            #if measured_phase - prev_phase > np.pi:
             #   phase_com_prev_phase = 2*np.pi - (measured_phase - prev_phase)
            #else:
            if -(3/4+0.1)*np.pi < measured_phase - prev_phase < (3/4+0.1)*np.pi:
                biased_phase = measured_phase
            else:
                if abs((measured_phase-prev_phase) - 2 * np.pi) < abs((measured_phase-prev_phase) + 2 * np.pi):
                    biased_phase = measured_phase - 2 * np.pi
                else:
                    biased_phase = measured_phase + 2 * np.pi

            phase_com_prev_phase = biased_phase - prev_phase

            # print("prev_phase", prev_phase)
            # print("measured phase", measured_phase)
            # print("biased_phase", biased_phase)
            # print("phase compared to prev_phase", phase_com_prev_phase)
            # print("delta phase for 00", phase_shift_zz)
            # print("delta phase for 01", phase_shift_zo)
            # print("delta phase for 10", phase_shift_oz)
            # print("delta phase for 11", phase_shift_oo)
            # closer to 00 then 01, 10 and 11
            if (abs(phase_com_prev_phase - phase_shift_zz) < abs(phase_com_prev_phase - phase_shift_zo)) and \
                    (abs(phase_com_prev_phase - phase_shift_zz) < abs(phase_com_prev_phase - phase_shift_oz)) and \
                    (abs(phase_com_prev_phase - phase_shift_zz) < abs(phase_com_prev_phase - phase_shift_oo)):
                data.append(0)
                data.append(0)
                prev_phase = measured_phase
            # closer to 01 then 00, 10 and 11
            elif (abs(phase_com_prev_phase - phase_shift_zo) < abs(phase_com_prev_phase - phase_shift_zz)) and \
                    (abs(phase_com_prev_phase - phase_shift_zo) < abs(phase_com_prev_phase - phase_shift_oz)) and \
                    (abs(phase_com_prev_phase - phase_shift_zo) < abs(phase_com_prev_phase - phase_shift_oo)):
                data.append(0)
                data.append(1)
                prev_phase = measured_phase
            # closer to 10 then 00, 01 and 11
            elif (abs(phase_com_prev_phase - phase_shift_oz) < abs(phase_com_prev_phase - phase_shift_zz)) and \
                    (abs(phase_com_prev_phase - phase_shift_oz) < abs(phase_com_prev_phase - phase_shift_zo)) and \
                    (abs(phase_com_prev_phase - phase_shift_oz) < abs(phase_com_prev_phase - phase_shift_oo)):
                data.append(1)
                data.append(0)
                prev_phase = measured_phase
            # closer to 11 then 00, 01 and 10
            else:
                data.append(1)
                data.append(1)
                prev_phase = measured_phase
        return self.decode_data(data)