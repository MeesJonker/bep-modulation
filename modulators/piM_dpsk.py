from modulation import Modulator
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi


class PiMDPSK(Modulator):
    def __init__(self, data_rate, carrier_frequency, carrier_amplitude, sampling_frequency, modulation_index, m, coding=False, j=3):
        super().__init__(data_rate, carrier_frequency, carrier_amplitude, sampling_frequency, modulation_index, m, coding=coding, j=j)
        self.bits_per_symbol = int(np.log2(m))
        self.detected_values = []
        self.m = m
        self.samples_per_symbol = int(self.bits_per_symbol * self.fs / self.data_rate)
        # define mapping table states according to the number of levels M
        if self.m == 2:
            states = [0, pi]
        elif self.m == 4:
            states = [-3*pi/4, -pi/4, pi/4, 3*pi/4]
        elif self.m == 8:
            states = [-7*pi/8, -5*pi/8, -pi/8, -3*pi/8, 7*pi/8, 5*pi/8, pi/8, 3*pi/8]
        elif self.m == 16:
            states = [-15*pi/16, -13*pi/16, -9*pi/16, -11*pi/16, -pi/16, -3*pi/16,
                      -7*pi/16, -5*pi/16, 15*pi/16, 13*pi/16, 9*pi/16, 11*pi/16,
                      pi/16, 3*pi/16, 7*pi/16, 5*pi/16]
        else:
            states = [-31*pi/32, -29*pi/32, -25*pi/32, -27*pi/32, -17*pi/32, -19*pi/32,
                      -23*pi/32, -21*pi/32, -pi/32, -3*pi/32, -7*pi/32, -5*pi/32, -15*pi/32,
                      -13*pi/32, -9*pi/32, -11*pi/32, 31*pi/32, 29*pi/32, 25*pi/32, 27*pi/32,
                      17*pi/32, 19*pi/32, 23*pi/32, 21*pi/32, pi/32, 3*pi/32, 7*pi/32, 5*pi/32,
                      15*pi/32, 13*pi/32, 9*pi/32, 11*pi/32]
        self.generate_mapping_table(states)

    def modulate(self, data):
        """Converts the given data bits to a pi/M DMPSK modulated signal usable for every m."""
        self.set_data(data)
        self.time = np.linspace(1/self.fs, (len(self.data_symbols)+1)*self.samples_per_symbol/self.fs, (len(self.data_symbols)+1)*self.samples_per_symbol)
        phase = [0]
        for symbol in self.data_symbols:
            phase.append(phase[-1] + self.mapping_table[tuple(symbol)])
        phase = np.repeat(np.array(phase), self.samples_per_symbol)

        self.signal = np.cos(2 * pi * self.carrier_frequency * self.time + phase)
        return self.signal

    def demodulate(self, signal):
        """Converts a pi/M DMPSK modulated signal to the data bits usable for m = 4,8,16,32."""
        self.time = np.linspace(1/self.fs, len(signal)/self.fs, len(signal))
        w = signal * np.cos(2 * pi * self.carrier_frequency * self.time)  # I arm
        z = -1*signal * np.sin(2 * pi * self.carrier_frequency * self.time)  # Q arm
        w = np.convolve(w, np.ones(self.samples_per_symbol))  # integrate for L (Tsym=2*Tb) duration
        z = np.convolve(z, np.ones(self.samples_per_symbol))  # integrate for L (Tsym=2*Tb) duration
        w = w[self.samples_per_symbol - 1::self.samples_per_symbol]  # I arm - sample at every symbol instant Tsym
        z = z[self.samples_per_symbol - 1::self.samples_per_symbol]  # Q arm - sample at every symbol instant Tsym
        theta_0 = np.angle(w + 1j*z)[:-1]
        theta_1 = np.angle(w + 1j*z)[1:]
        d_theta = theta_1 - theta_0
        for i in range(len(d_theta)):
            if d_theta[i] > pi:
                d_theta[i] -= 2*pi
            elif d_theta[i] < -pi:
                d_theta[i] += 2*pi

        data = []
        d_theta_mapped = np.zeros(len(d_theta))
        mapping_values = np.array([v for k, v in self.mapping_table.items()])
        for i, val in enumerate(d_theta):
            d_theta_mapped[i] = mapping_values[np.argmin(abs(mapping_values-val))]
        for diff in d_theta_mapped:
            data.extend(list(self.demapping_table[diff]))
        data = np.array(data)
        return self.decode_data(data)

    def iq_plot(self,U,V):
        #p = plt.figure(figsize=(6, 6))
        plt.plot(U, V, 'o', markersize=1)
        for i in range(0, self.samples_per_symbol * 4, int(self.samples_per_symbol)):
            plt.plot(U[i], V[i], 'o', markersize = 5, label = 'no %i'%(i/self.samples_per_symbol))

        plt.xlabel("In-phase Amplitude")
        plt.ylabel("Quadrature Amplitude")
        plt.title('u,v')
        plt.legend()
        plt.tight_layout()
        plt.show()
        return 0


if __name__ == '__main__':
    # Modulation parameters
    datarate = 10000                # data rate [bits/s]
    carrier_freq = 50000            # carrier frequency [Hz]
    carrier_ampl = 1                # carrier amplitude
    fs = 500000                     # ADC/DAC sampling frequency [Hz]
    mod_index = 1                   # modulation index

    eb_n0_db = 40

    number_of_bits = 42000          # number of bits to be transmitted
    m = 32                           # number of levels

    mod = PiMDPSK(datarate, carrier_freq, carrier_ampl, fs, mod_index, m, True, 3)  # modulator
    bits = np.random.randint(2, size=number_of_bits)    # array of data bits to be sent
    #bits = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1])
    signal = mod.modulate(bits)
    pwr = mod.signal_power()
    eb_n0 = 10**(0.1*eb_n0_db)                      # Eb/N0 linear
    noise_power_density = (mod.signal_power()/datarate)/eb_n0    # [V^2/Hz]
    noise = np.random.normal(0, np.sqrt(fs/2*noise_power_density), len(signal))
    decoded = mod.demodulate(signal)
    if len(bits) % int(np.log2(m)) != 0:
        bits_not_used = len(bits) % int(np.log2(m))
        bits = bits[0: -1 * bits_not_used]
    signal_plus_noise, ber = mod.bit_error_rate(bits, noise)
    print('BER:', ber)
    plt.plot(mod.time[0:5 * mod.samples_per_symbol] * 1000000, signal[0:5 * mod.samples_per_symbol], label='Transmitted')
    plt.xlabel('Time [ms]')
    plt.ylabel('Amplitude')
    plt.title('Transmitted and received signal')
    plt.tight_layout()
    plt.legend()
    plt.show()