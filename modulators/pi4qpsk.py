from modulation import Modulator
import numpy as np
import matplotlib.pyplot as plt

# Pi/4-QPSK implementation largely adopted from the book:
# Digital Modulations using Python by Mathuranathan Viswanathan
class Pi4QPSK(Modulator):
    def __init__(self, data_rate, carrier_frequency, carrier_amplitude, sampling_frequency, modulation_index, M):
        super().__init__(data_rate, carrier_frequency, carrier_amplitude, sampling_frequency, modulation_index, M)
        self.bits_per_symbol = 2
        self.detected_values = []
        self.samples_per_symbol = int(2 * self.fs / self.data_rate)

    def modulate(self, data):
        """Converts the given data bits to a pi/4 DQPSK modulated signal."""
        I = data[0::2]  # odd bit stream
        Q = data[1::2]  # even bit stream
        # club 2-bits to form a symbol and use it as index for dTheta table
        m = 2 * I + Q
        dTheta = np.array([-3 * np.pi / 4, 3 * np.pi / 4, -np.pi / 4, np.pi / 4])  # LUT for pi/4-DQPSK
        u = np.zeros(len(m) + 1)
        v = np.zeros(len(m) + 1)
        u[0] = 1
        v[0] = 0  # initial conditions for uk and vk
        for k in range(0, len(m)):
            u[k + 1] = u[k] * np.cos(dTheta[m[k]]) - v[k] * np.sin(dTheta[m[k]])
            v[k + 1] = u[k] * np.sin(dTheta[m[k]]) + v[k] * np.cos(dTheta[m[k]])

        # Waveform formation (similar to conventional QPSK)
        U = np.tile(u, (self.samples_per_symbol, 1)).flatten('F')  # odd bit stream at 1/2Tb baud
        V = np.tile(v, (self.samples_per_symbol, 1)).flatten('F')  # even bit stream at 1/2Tb baud
        self.time = np.linspace(1/self.fs, len(U)/self.fs, len(U))  # time base
        U_t = U * np.cos(2 * np.pi * self.carrier_frequency * self.time)
        V_t = -V * np.sin(2 * np.pi * self.carrier_frequency * self.time)
        self.signal = U_t + V_t
        return self.signal

    def demodulate(self, signal):
        self.time = np.linspace(1/self.fs, len(signal)/self.fs, len(signal))
        w = signal * np.cos(2 * np.pi * self.carrier_frequency * self.time)  # I arm
        z = -signal * np.sin(2 * np.pi * self.carrier_frequency * self.time)  # Q arm
        w = np.convolve(w, np.ones(self.samples_per_symbol))  # integrate for L (Tsym=2*Tb) duration
        z = np.convolve(z, np.ones(self.samples_per_symbol))  # integrate for L (Tsym=2*Tb) duration
        w = w[self.samples_per_symbol - 1::self.samples_per_symbol]  # I arm - sample at every symbol instant Tsym
        z = z[self.samples_per_symbol - 1::self.samples_per_symbol]  # Q arm - sample at every symbol instant Tsym

        x = np.zeros(len(w) - 1)
        y = np.zeros(len(w) - 1)
        for k in range(0, len(w) - 1):
            x[k] = w[k + 1] * w[k] + z[k + 1] * z[k]
            y[k] = z[k + 1] * w[k] - w[k + 1] * z[k]
        data = np.zeros(2 * len(x))
        data[0::2] = (x > 0)  # odd bits
        data[1::2] = (y > 0)  # even bits
        return self.decode_data(data)


if __name__ == '__main__':
    # Modulation parameters
    datarate = 8000                # data rate [bits/s]
    carrier_freq = 50000            # carrier frequency [Hz]
    carrier_ampl = 1                # carrier amplitude
    fs = 500000                     # ADC/DAC sampling frequency [Hz]
    mod_index = 1                   # modulation index

    eb_n0_db = 20

    number_of_bits = 1000          # number of bits to be transmitted
    m = 4                           # number of levels

    mod = Pi4QPhaseShiftKeying(datarate, carrier_freq, carrier_ampl, fs, mod_index, m)  # modulator
    bits = np.random.randint(2, size=number_of_bits)    # array of data bits to be sent

    signal = mod.modulate(bits)
    pwr = mod.signal_power()
    eb_n0 = 10**(0.1*eb_n0_db)                      # Eb/N0 linear
    noise_power_density = (mod.signal_power()/datarate)*fs/2/eb_n0    # [V^2/Hz]
    noise = np.random.normal(0, np.sqrt(noise_power_density), len(signal))
    decoded = mod.demodulate(signal)

    signal_plus_noise, ber = mod.bit_error_rate(bits, noise)

    print('BER:', ber)
    # plot signal for first 10 symbols
    #print(pwr)
    #plt.plot(hilb[0:100000])
    plt.plot(mod.time[0:10*mod.samples_per_symbol]*1000000, decoded[0:10*mod.samples_per_symbol])
    plt.xlabel('Time [ms]')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()