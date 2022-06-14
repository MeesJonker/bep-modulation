from modulation import Modulator
import numpy as np
import matplotlib.pyplot as plt


# QAM implementation largely adopted from the book:
# Digital Modulations using Python by Mathuranathan Viswanathan
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
            self.m = M  # number of states

        n = np.arange(0, self.M)  # Sequential address from 0 to M-1 (1xM dimension)
        a = np.asarray([x ^ (x >> 1) for x in n])  # convert linear addresses to Gray code
        D = np.sqrt(M).astype(int)  # Dimension of K-Map - N x N matrix
        a = np.reshape(a, (D, D))  # NxN gray coded matrix
        oddRows = np.arange(start=1, stop=D, step=2)  # identify alternate rows
        a[oddRows, :] = np.fliplr(a[oddRows, :])  # Flip rows - KMap representation
        nGray = np.reshape(a, (self.M))  # reshape to 1xM - Gray code walk on KMap
        # Construction of ideal M-QAM constellation from sqrt(M)-PAM
        (x, y) = np.divmod(nGray, D)  # element-wise quotient and remainder
        Ax = 2 * x + 1 - D  # PAM Amplitudes 2d+1-D - real axis
        Ay = 2 * y + 1 - D  # PAM Amplitudes 2d+1-D - imag axis
        states = Ax + 1j * Ay
        self.generate_mapping_table(states)

    def modulate(self, data):
        """Converts the given data bits to a QAM modulated signal."""
        self.set_data(data)

        # define amplitude and phase vectors given the data symbols
        iq_points = np.array([self.mapping_table[tuple(symbol)] for symbol in
                              self.data_symbols])
        iq_points = np.repeat(iq_points, self.samples_per_symbol)

        # create signal
        self.signal = np.real(iq_points) * np.cos(2 * np.pi * self.carrier_frequency * self.time) \
                      - np.imag(iq_points) * np.sin(2 * np.pi * self.carrier_frequency * self.time)
        return self.signal

    def demodulate(self, signal):
        """Demodulates the given QAM signal to retrieve the data bits."""
        # filter out noise and determine complex envelope using the hilbert transform
        #signal = temporal_bandpass_filter(signal, self.fs, 0.9 * self.carrier_frequency,
        #                                  1.1 * self.carrier_frequency)
        i = np.convolve(signal * np.cos(2 * np.pi * self.carrier_frequency * self.time),
                        np.ones(self.samples_per_symbol))
        q = np.convolve(signal * np.sin(2 * np.pi * self.carrier_frequency * self.time),
                        np.ones(self.samples_per_symbol))
        received_iq = i - 1j * q
        self.received_symbols = received_iq[self.samples_per_symbol - 1::self.samples_per_symbol]
        self.received_symbols *= (np.max(np.abs([v for k, v in self.mapping_table.items()]))/np.max(np.abs(self.received_symbols)))
        self.detected_symbols = self.iq_mapping()
        data = []
        for symbol in self.detected_symbols:
            bits_detected = list(self.demapping_table[symbol])
            data.extend(bits_detected)
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
        boundaries = [a * dist_between_dots for a in range(-int(np.sqrt(self.m) - 1) // 2,
                                                           int(np.sqrt(self.m) - 1) // 2 + 1)]
        plt.vlines(boundaries, ymin=-9, ymax=9, color='k', linestyles='--', linewidth=1)
        plt.hlines(boundaries, xmin=-9, xmax=9, color='k', linestyles='--', linewidth=1)
        plt.axhline(y=0, color='k')
        lim = dist_between_dots * np.sqrt(self.m) / 2
        plt.ylim([-lim, lim])
        plt.xlim([-lim, lim])
        plt.xlabel("In-phase Amplitude")
        plt.ylabel("Quadrature Amplitude")
        plt.title(f"IQ plot {self.m}-QAM")
        # plt.show()
        return p


if __name__ == '__main__':
    # Modulation parameters
    datarate = 10000  # data rate [bits/s]
    carrier_freq = 50000  # carrier frequency [Hz]
    carrier_ampl = 1  # carrier amplitude
    fs = 1000000  # ADC/DAC sampling frequency [Hz]
    mod_index = 1  # modulation index

    eb_n0_db = 0

    number_of_bits = 3000  # number of bits to be transmitted
    m = 16  # number of levels
    bits = np.random.randint(2, size=number_of_bits)    # array of data bits to be sent

    mod = QAM(datarate, carrier_freq, carrier_ampl, fs, mod_index, m)
    signal = mod.modulate(bits)
    demod = mod.demodulate(signal)
    print(bits[:20], '\n', np.array(demod)[:20])