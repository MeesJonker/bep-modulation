import numpy as np
import matplotlib.pyplot as plt
from waterfilling import divide_spectrum, power_allocation, sort_steps, rearrange_powers
from modulators.piM_DPSK import PiMDPSK
from ber_controller import create_modulators, get_curves, ber_controller
from bandpass import temporal_bandpass_filter


# Old implementation

def encode_data(modulators, data, return_psd=True):
    signals = []
    for modulator in modulators:
        signals.append(modulator.modulate(data))
    signal = np.zeros(len(signals[0]))

    for s in signals:
        signal += s
    psds = []
    if return_psd:
        for s in signals:
            psd = np.abs(np.fft.fft(s))[:int(len(s)/2)]
            psd /= np.max(psd)
            psds.append(psd)
        return signal, psds
    else:
        return signal


def decode_signal(modulators, carrier_frequencies, signal):
    data = []
    for mod in modulators:
        data.append(mod.demodulate(signal))
    return data


def ber(data_in, data_out):
    bers = []
    for data in data_out:
        num_errors = np.sum(np.abs(np.subtract(np.array(data_in), np.array(data))))
        bers.append(num_errors / len(data))  # bit error rate
    return bers


class Multicarrier:
    """Implements multi-carrier modulation by combining the single-carrier
       modulator classes"""
    def __init__(self, bandwidth, num_carriers, symbol_rate, fs, power_budget, frequency_axis, noise_spectrum, transfer_function):
        self.bandwidth = bandwidth
        self.num_carriers = num_carriers
        self.symbol_rate = symbol_rate
        self.fs = fs
        self.mod = []

        self.power_budget = power_budget
        self.f = frequency_axis
        self.n = noise_spectrum
        self.h = transfer_function
        self.subcarrier_powers()
        self.subcarrier_modulators()
        self.subcarrier_bandwidth = bandwidth/num_carriers

        self.modulators = []
        self.rates = []
        filenames = ['plotters/data/pi4dpsk_j5.csv', 'plotters/data/pi4dpsk.csv',
                     'plotters/data/pi8dpsk_j5.csv', 'plotters/data/pi8dpsk.csv',
                     'plotters/data/pi16dpsk_j5.csv', 'plotters/data/pi16dpsk.csv',
                     'plotters/data/pi32dpsk_j5.csv', 'plotters/data/pi32dpsk.csv']
        self.curves = get_curves(filenames)

    def subcarrier_powers(self):
        self.channel_noise_ratio = self.h/self.n
        inverse_cnr = 1/self.channel_noise_ratio
        self.f_steps, self.inverse_cnr_steps = divide_spectrum(self.f, inverse_cnr, self.num_carriers)
        sorted_step_depths, indices = sort_steps(self.inverse_cnr_steps)
        self.power_levels = power_allocation(self.power_budget, sorted_step_depths)
        self.power_levels = rearrange_powers(self.power_levels, indices)
        #self.power_levels = self.power_budget/self.num_carriers*np.ones(self.num_carriers)

    def subcarrier_modulators(self):
        for k, fr in enumerate(self.f_steps):
            carrier_amplitude = np.sqrt(2 * self.power_levels[k])
            self.mod.append(create_modulators(self.symbol_rate, fr, carrier_amplitude, self.fs))

    def modulate_packet(self, data, eb_n0_per_band, packet_duration):
        mods = []
        data_per_carrier = []
        s = []
        for j, eb_n0 in enumerate(eb_n0_per_band):
            mod_idx = ber_controller(eb_n0, 10**(-6), self.curves)  # select modulation level
            mods.append(self.mod[j][mod_idx])
        for j, mod in enumerate(mods):
            effective_rate = mod.data_rate
            if mod.coding:
                effective_rate *= (26/31)
            data_size = int(effective_rate*packet_duration)//1560 * 1560
            if len(data) >= data_size:
                subcarrier_data = data[:data_size]
                data_per_carrier.append(subcarrier_data)
                s.append(mod.modulate(subcarrier_data))
                data = data[data_size:]
        signal = np.zeros(np.max([len(s_i) for s_i in s]))
        for j, s_i in enumerate(s):
            signal[:len(s_i)] += s_i
            signal_power = mods[j].carrier_amplitude**2/2
            bit_energy = signal_power/mods[j].data_rate
            n0 = bit_energy/eb_n0_per_band[j]
            noise = np.sqrt(n0 * mods[j].fs / 2) * np.random.standard_normal(len(s_i))
            carrier_spacing = self.bandwidth/self.num_carriers
            in_band_noise = temporal_bandpass_filter(noise, mods[j].fs, mods[j].carrier_frequency-carrier_spacing/2, mods[j].carrier_frequency+carrier_spacing/2)
            plt.plot(np.fft.fft(in_band_noise))
            signal[:len(s_i)] += in_band_noise
        plt.title('noise')
        plt.show()


        return signal, data, data_per_carrier, mods

    def demodulate_packet(self, signal, mods):
        data_per_carrier = []
        for mod in mods:
            detected_data = mod.demodulate(signal)
            data_per_carrier.append(detected_data)
        return data_per_carrier


if __name__ == '__main__':
    datarate = 50000                # data rate [bits/s]
    fs = 1000000                    # ADC/DAC sampling frequency [Hz]
    mod_index = 1                   # modulation index
    m = 32
    P = 50
    number_of_bits = 20000

    total_bandwidth = 200000
    num_carriers = 20

    f = np.linspace(0, total_bandwidth, 2*total_bandwidth+1)
    h = np.abs(1/(1 + 1j*(f**2)/20000/10**6))
    spec = 1.5+0.05*np.random.randn(len(f)) + 10000/(f+2000) + f**2/10**6

    bits = np.random.randint(2, size=number_of_bits)  # array of data bits to be sent


    bandwidth_per_carrier = total_bandwidth/num_carriers
    carrier_frequencies = np.linspace(bandwidth_per_carrier/2, total_bandwidth-bandwidth_per_carrier/2, num_carriers)
    f_bands, noise_bands = divide_spectrum(f, spec, num_carriers)
    f_bands, h_bands = divide_spectrum(f, h, num_carriers)
    inverse_cnr = 1 / h_bands
    power_levels = power_allocation(P, inverse_cnr)

    modulators = []
    for i, f in enumerate(carrier_frequencies):
        carrier_amplitude = np.sqrt(2*power_levels[i])
        modulators.append(PiMDPSK(datarate, f, carrier_amplitude, fs, mod_index, m))

    signal, psds = encode_data(modulators, bits)
    signal = np.clip(signal, -20, 20)
    signal_psd = np.abs(np.fft.fft(signal))
    signal_psd /= np.max(signal_psd)
    f_axis = np.linspace(0, modulators[0].fs/2, int(len(signal)/2))
    plt.figure(figsize=(6, 3))
    for psd in psds:
        plt.plot(f_axis/1000, 10*np.log10(psd), alpha=0.5)
    plt.xlim([0, 200])
    plt.title("PSD of OFDM-carriers")
    plt.xlabel("Frequency [kHz]")
    plt.ylabel("Signal energy [dB]")
    plt.ylim([-20, 5])
    plt.tight_layout()
    plt.savefig("plotters/plots/multicarrier.pdf", bbox_inches='tight', dpi=600)
    plt.show()
    data_decoded = decode_signal(modulators, carrier_frequencies, signal)

    bit_errors = ber(bits, data_decoded)
    print(bit_errors)


