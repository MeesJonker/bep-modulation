from modulation import Modulator
import numpy as np
import matplotlib.pyplot as plt


class DPSK(Modulator):
    def __init__(self, data_rate, carrier_frequency, carrier_amplitude, sampling_frequency,
                 modulation_index, M, coding=False, j=5):
        super().__init__(data_rate, carrier_frequency, carrier_amplitude, sampling_frequency,
                         modulation_index, 2, coding=coding, j=j)

    def modulate(self, data):
        """Converts the given data bits to a DPSK modulated signal."""
        self.set_data(data)
        self.samples_per_symbol = int(self.fs / self.data_rate)
        self.signal_length = int((1+len(self.data_symbols)) * self.samples_per_symbol)
        self.time = np.linspace(1 / self.fs, self.signal_length / self.fs, self.signal_length)
        phase = []
        encoded_bit = 0

        # differentially encode bit stream
        phase.append(encoded_bit)
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
        # multiply signal with in-phase and quadrature reference
        p = signal * np.cos(2 * np.pi * self.carrier_frequency * self.time)  # multiply I arm by cos
        q = signal * np.sin(2 * np.pi * self.carrier_frequency * self.time)  # multiply Q arm by sin
        # integrate the result over a symbol period
        x = np.convolve(p, np.ones(self.samples_per_symbol))  # integrate I-arm by Tb duration (L samples)
        y = np.convolve(q, np.ones(self.samples_per_symbol))  # integrate Q-arm by Tb duration (L samples)
        # sample from the result
        xk = x[self.samples_per_symbol - 1::self.samples_per_symbol]  # Sample every Lth sample
        yk = y[self.samples_per_symbol - 1::self.samples_per_symbol]  # Sample every Lth sample

        w0 = xk[0:-1]  # non delayed version on I-arm
        w1 = xk[1:]  # 1 bit delay on I-arm
        z0 = yk[0:-1]  # non delayed version on Q-arm
        z1 = yk[1:]  # 1 bit delay on Q-arm

        # obtain bits from angle differences
        ang0 = np.angle(w0 + 1j*z0)
        ang1 = np.angle(w1 + 1j*z1)
        ang_diff = ang1-ang0
        ang_diff[ang_diff > 1.5*np.pi] -= 2*np.pi
        ang_diff[ang_diff < -1.5 * np.pi] += 2 * np.pi
        data = (np.abs(ang_diff) > np.pi/2)  # threshold detection
        return self.decode_data(data)


if __name__ == '__main__':
    # Modulation parameters
    datarate = 10000                # data rate [bits/s]
    carrier_freq = 50000            # carrier frequency [Hz]
    carrier_ampl = 1                # carrier amplitude
    fs = 500000                     # ADC/DAC sampling frequency [Hz]
    mod_index = 1                   # modulation index

    eb_n0_db = 0

    number_of_bits = 1000           # number of bits to be transmitted
    m = 2                           # number of levels

    mod = DPSK(datarate, carrier_freq, carrier_ampl, fs, mod_index, m)  # modulator
    bits = np.random.randint(2, size=number_of_bits)    # array of data bits to be sent

    signal = mod.modulate(bits)
    pwr = mod.signal_power()
    eb_n0 = 10**(0.1*eb_n0_db)                      # Eb/N0 linear
    noise_power_density = (mod.signal_power()/datarate)/eb_n0    # [V^2/Hz]
    noise = np.random.normal(0, np.sqrt(noise_power_density), len(signal))
    decoded = mod.demodulate(signal)

    signal_plus_noise, ber = mod.bit_error_rate(bits, noise)

