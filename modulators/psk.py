from modulation import Modulator
import numpy as np
import matplotlib.pyplot as plt
from raised_cosine import raised_cosine


class PSK(Modulator):
    def __init__(self, data_rate, carrier_frequency, carrier_amplitude, sampling_frequency,
                 modulation_index, M):
        super().__init__(data_rate, carrier_frequency, carrier_amplitude, sampling_frequency,
                         modulation_index, M)

    def modulate(self, data):
        """Converts the given data bits to a PSK modulated signal."""
        self.set_data(data)

        # set phase levels corresponding to each data symbol and generate mapping table
        m = np.arange(0, self.M)  # all information symbols m={0,1,...,M-1}
        I = 1 / np.sqrt(2) * np.cos(m / self.M * 2 * np.pi)
        Q = 1 / np.sqrt(2) * np.sin(m / self.M * 2 * np.pi)
        states = I - 1j * Q
        self.generate_mapping_table(states)

        # create phase vector given the symbols to be transmitted
        iq = [self.mapping_table[tuple(symbol)] for symbol in self.data_symbols]
        # extend the phase vector to cover all signal samples
        iq = np.repeat(iq, self.samples_per_symbol)

        # construct the signal
        signal_i = np.real(iq) * np.cos(2 * np.pi * self.carrier_frequency * self.time)
        signal_q = np.imag(iq) * np.sin(2 * np.pi * self.carrier_frequency * self.time)
        self.signal = signal_i + signal_q
        return self.signal

    def demodulate(self, signal):
        """Demodulates the given PSK signal to retrieve the data bits"""
        i = np.convolve(signal * np.cos(2 * np.pi * self.carrier_frequency * self.time),
                        np.ones(self.samples_per_symbol))
        q = np.convolve(signal * -np.sin(2 * np.pi * self.carrier_frequency * self.time),
                        np.ones(self.samples_per_symbol))
        received_iq = i - 1j * q
        self.received_symbols = received_iq[self.samples_per_symbol - 1::self.samples_per_symbol]
        self.received_symbols *= (
                    np.max(np.abs([v for k, v in self.mapping_table.items()])) / np.max(np.abs(self.received_symbols)))
        self.detected_symbols = self.iq_mapping()
        data = []
        for symbol in self.detected_symbols:
            bits_detected = list(self.demapping_table[symbol])
            data.extend(bits_detected)
        return self.decode_data(data)


if __name__ == '__main__':
    # Modulation parameters
    datarate = 20000  # data rate [bits/s]
    carrier_freq = 50000  # carrier frequency [Hz]
    carrier_ampl = 1  # carrier amplitude
    fs = 1000000  # ADC/DAC sampling frequency [Hz]
    mod_index = 1  # modulation index

    eb_n0_db = 0

    number_of_bits = 10000  # number of bits to be transmitted
    m = 4  # number of levels
    bits = np.random.randint(2, size=number_of_bits)  # array of data bits to be sent

    mod = PSK(datarate, carrier_freq, carrier_ampl, fs, mod_index, m)
    signal = mod.modulate(bits)
    width = 20
    demod = mod.demodulate(signal)
    signal_noise, ber = mod.bit_error_rate(bits, np.zeros(len(signal)))
    print(ber)
    plt.plot(signal[100:5000])
    plt.show()
    print(bits[:20], '\n', np.array(demod)[:20])
