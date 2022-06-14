from modulation import Modulator
import numpy as np


class ASK(Modulator):
    def __init__(self, data_rate, carrier_frequency, carrier_amplitude, sampling_frequency,
                 modulation_index, M):
        super().__init__(data_rate, carrier_frequency, carrier_amplitude, sampling_frequency,
                         modulation_index, M)

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
        self.signal = amplitude * np.cos(2 * np.pi * self.carrier_frequency * self.time)
        return self.signal

    def demodulate(self, signal, coherent=True):
        """Demodulates the given ASK signal to retrieve the data bits"""
        if coherent:
            reference_signal = np.cos(2 * np.pi * self.carrier_frequency * self.time)
        else:
            # for non-coherent demodulation, the reference signal is not in phase
            # with the transmitter oscillator
            reference_signal = np.cos(2 * np.pi * self.carrier_frequency * self.time + np.pi)

        # multiply by reference and integrate
        conv = np.convolve(signal * reference_signal, np.ones(self.samples_per_symbol))
        # normalize signal
        conv *= np.max([v for k, v in self.mapping_table.items()])/np.max(conv)

        # detect symbols
        self.received_symbols = conv[self.samples_per_symbol - 1::self.samples_per_symbol]
        self.detected_symbols = self.iq_mapping()
        data = []
        for symbol in self.detected_symbols:
            bits_detected = list(self.demapping_table[symbol])
            data.extend(bits_detected)
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
    m = 4  # number of levels
    bits = np.random.randint(2, size=number_of_bits)  # array of data bits to be sent

    mod = ASK(datarate, carrier_freq, carrier_ampl, fs, mod_index, m)
    signal = mod.modulate(bits)
    demod = mod.demodulate(signal)

    print(bits[:20], '\n', np.array(demod)[:20])