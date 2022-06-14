import numpy as np
from plotters.eb_n0 import get_averages
from modulators.piM_DPSK import PiMDPSK


def ber_controller(eb_n0_lin, max_ber, curves):
    """Controls the modulation and coding schemes based on Eb/N0 values,
    such that the BER is kept below a certain maximum"""
    # initialize thresholds
    thresholds = []
    # loop over all given BER curves
    for curve in curves:
        # obtain Eb/N0 threshold at which the BER is sufficiently low for the given curve
        thresholds.append(-5+0.5*next(idx for idx, val in enumerate(curve) if val <= max_ber))

    # select the technique which has a sufficiently low BER for the given Eb/N0 value
    threshold_index = next(idx for idx, val in enumerate(thresholds) if (val > 10*np.log10(eb_n0_lin)) or idx == len(thresholds)-1)
    if threshold_index == 0:
        # in this case, Eb/N0 is too low to maintain a sufficiently low BER
        # the lowest order modulation is selected
        return 0
    elif threshold_index == len(thresholds)-1:
        return threshold_index
    else:
        return threshold_index - 1


def create_modulators(symbol_rate, carrier_freq, carrier_ampl, fs):
    """Generates a dictionary of modulators with varying number of levels M and coding
       at the given carrier frequency and symbol rate."""
    mod_dict = {0: PiMDPSK(2 * symbol_rate, carrier_freq, carrier_ampl, fs, 1, 4, coding=True, j=5),
                1: PiMDPSK(2 * symbol_rate, carrier_freq, carrier_ampl, fs, 1, 4, coding=False),
                2: PiMDPSK(3 * symbol_rate, carrier_freq, carrier_ampl, fs, 1, 8, coding=True, j=5),
                3: PiMDPSK(3 * symbol_rate, carrier_freq, carrier_ampl, fs, 1, 8, coding=False),
                4: PiMDPSK(4 * symbol_rate, carrier_freq, carrier_ampl, fs, 1, 16, coding=True, j=5),
                5: PiMDPSK(4 * symbol_rate, carrier_freq, carrier_ampl, fs, 1, 16, coding=False),
                6: PiMDPSK(5 * symbol_rate, carrier_freq, carrier_ampl, fs, 1, 32, coding=True, j=5),
                7: PiMDPSK(5 * symbol_rate, carrier_freq, carrier_ampl, fs, 1, 32, coding=False)}
    return mod_dict


def get_curves(filenames):
    """Retrieves BER curves from given files."""
    curves = np.zeros((len(filenames), 81))
    for i, name in enumerate(filenames):
        curve = np.array(get_averages(name))
        curves[i][:len(curve)] = curve
    return curves


if __name__ == '__main__':
    min_ber = 10**(-6)
    # obtain the BER vs. Eb/N0 curves
    filenames = ['plotters/data/pi4dpsk_j5.csv', 'plotters/data/pi4dpsk.csv',
                 'plotters/data/pi8dpsk_j5.csv', 'plotters/data/pi8dpsk.csv',
                 'plotters/data/pi16dpsk_j5.csv', 'plotters/data/pi16dpsk.csv',
                 'plotters/data/pi32dpsk_j5.csv', 'plotters/data/pi32dpsk.csv']
    curves = get_curves(filenames)

    # get modulator index
    idx = ber_controller(1000, min_ber, curves)

    # define modulators in dictionary with equal symbol rate
    symbol_rate = 10000
    carrier_freq = 50000
    carrier_ampl = 1
    fs = 1000000
    modulators = dict()
    modulators[carrier_freq] = create_modulators(symbol_rate, carrier_freq, carrier_ampl, fs)
    # check if the correct modulator was selected
    mod = modulators[carrier_freq][idx]
    print(mod.M)








