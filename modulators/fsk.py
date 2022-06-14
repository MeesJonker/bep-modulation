from modulation import Modulator
import numpy as np
from bandpass import temporal_bandpass_filter
from scipy.signal import hilbert


class FSK(Modulator):
    def __init__(self, data_rate, carrier_frequency, carrier_amplitude, sampling_frequency,
                 modulation_index, M, coherent=True):
        super().__init__(data_rate, carrier_frequency, carrier_amplitude, sampling_frequency,
                         modulation_index, M)
        self.coherent = coherent
        # set frequency levels
        self.levels = np.concatenate((np.linspace(-self.M / 2 + 0.5, -0.5, int(self.M / 2)),
                                 np.linspace(0.5, self.M / 2 - 0.5, int(self.M / 2))))
        if self.M == 2:
            self.levels = np.array([-0.5, 0.5])

    def modulate(self, data):
        """Converts the given data bits to an FSK modulated signal."""
        self.set_data(data)

        # generate mapping table
        states = self.carrier_frequency + self.modulation_index * self.symbol_rate * self.levels
        self.generate_mapping_table(states)

        # create frequency vector given the symbols to be transmitted
        frequency = [self.mapping_table[tuple(symbol)] for symbol in self.data_symbols]
        # extend the frequency vector to cover all signal samples
        frequency = np.repeat(frequency, self.samples_per_symbol)

        # construct the signal
        self.signal = self.carrier_amplitude * np.cos(2 * np.pi * frequency * self.time)
        return self.signal

    def demodulate(self, signal):
        """Demodulates the given FSK signal to retrieve the data bits"""
        if not self.coherent:
            frequencies = self.carrier_frequency+self.modulation_index*self.symbol_rate*self.levels
            # find spacing between adjacent frequency levels
            spacing = abs(frequencies[1] - frequencies[0])

            # create arrays to store bandpass-filtered copies of the signal and their envelopes
            filtered_signals = np.ndarray((self.M, len(signal)))
            envelopes = np.ndarray((self.M, len(signal)))
            integrated_envelopes = np.ndarray((self.M, len(signal)+self.samples_per_symbol-1))

            # apply bandpass filters to discriminate between frequencies in the signal
            for i, freq in enumerate(frequencies):
                filtered_signals[i] = temporal_bandpass_filter(signal, self.fs, freq - 0.5 *
                                                               spacing, freq + 0.5 * spacing)

                envelopes[i] = np.abs(hilbert(filtered_signals[i]))
                integrated_envelopes[i] = np.convolve(envelopes[i], np.ones(self.samples_per_symbol))

            # find number of bits (assuming the entire signal contains useful bits)
            num_symbols = (int(len(signal) - self.samples_per_symbol / 2) // self.samples_per_symbol) + 1

            # derive de-mapping table to map frequencies to back to symbols
            demapping_table = {v: k for k, v in self.mapping_table.items()}
            # find data bits by sampling the frequency in the middle of the signaling interval
            data = []
            for i in range(num_symbols):
                amplitudes = integrated_envelopes[:, int(self.samples_per_symbol-1) + i *
                                       self.samples_per_symbol]
                freq = frequencies[np.argmax(amplitudes, axis=0)]
                data.extend(list(demapping_table[freq]))
        else:
            convolutions = np.zeros((self.M, len(signal)+self.samples_per_symbol-1))
            frequencies = [v for k, v in self.mapping_table.items()]
            for i in range(self.M):
                convolutions[i] = np.convolve(signal * np.cos(2*np.pi*frequencies[i]*self.time), np.ones(self.samples_per_symbol))
            samples = convolutions[:, self.samples_per_symbol-1::self.samples_per_symbol]
            max_indices = np.argmax(samples, axis=0)
            data = []
            for i in max_indices:
                data.extend(list(self.demapping_table[frequencies[i]]))
        return self.decode_data(np.array(data))


if __name__ == '__main__':
    # Modulation parameters
    datarate = 10000  # data rate [bits/s]
    carrier_freq = 50000  # carrier frequency [Hz]
    carrier_ampl = 1  # carrier amplitude
    fs = 1000000  # ADC/DAC sampling frequency [Hz]
    mod_index = 1  # modulation index

    eb_n0_db = 0

    number_of_bits = 3000  # number of bits to be transmitted
    m = 2  # number of levels
    bits = np.random.randint(2, size=number_of_bits)    # array of data bits to be sent

    mod = FSK(datarate, carrier_freq, carrier_ampl, fs, mod_index, m)
    signal = mod.modulate(bits)
    demod = mod.demodulate(signal)