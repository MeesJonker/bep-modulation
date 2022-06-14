import numpy as np
import matplotlib.pyplot as plt
from ber_controller import ber_controller, get_curves

# define a range of channel capacity values and compute Eb/N0
cap = np.arange(0.001, 20, 0.001)
eb_n0_lin = (2 ** cap - 1) / cap
eb_n0_db = 10 * np.log10(eb_n0_lin)

# retrieve the BER curves obtained with simulations
filenames = ['data/pi4dpsk_j5.csv', 'data/pi4dpsk.csv',
             'data/pi8dpsk_j5.csv', 'data/pi8dpsk.csv',
             'data/pi16dpsk_j5.csv', 'data/pi16dpsk.csv',
             'data/pi32dpsk_j5.csv', 'data/pi32dpsk.csv']
curves = get_curves(filenames)

eff_levels = [1.6774, 2, 2.5161, 3, 3.3548, 4, 4.1935, 5]
spectral_efficiencies6 = []
spectral_efficiencies3 = []

# define spectral efficiencies over the range of Eb/N0 values
# for BER < 10e-6 and BER < 10e-3
for eb_n0 in eb_n0_lin:
    idx6 = ber_controller(eb_n0, 10 ** (-6), curves)
    idx3 = ber_controller(eb_n0, 10 ** (-3), curves)
    if eb_n0 > 11.6:
        spectral_efficiencies6.append(eff_levels[idx6])
        spectral_efficiencies3.append(eff_levels[idx3])
    else:
        spectral_efficiencies6.append(0)
        spectral_efficiencies3.append(0)
# compute the ratio between the data rate and capacity
ratio6 = np.array(spectral_efficiencies6) / cap
ratio3 = np.array(spectral_efficiencies3) / cap

# generate figure
fig, axs = plt.subplots(2, figsize=(6, 4))
plt.suptitle("Capacity utilization of adaptive system")
axs[0].plot(eb_n0_db, cap)
axs[0].text(s='Shannon limit: \n$E_b/N_0$ = -1.59 dB', x=-1.3, y=5)
axs[0].plot(eb_n0_db, spectral_efficiencies6)
axs[0].plot(eb_n0_db, spectral_efficiencies3, color='orange', linestyle='--', alpha=0.4)
axs[0].axvline(x=-1.59, linestyle='--', alpha=0.5, label='axvline - full height')
axs[0].set_yscale('log')
axs[0].set_ylim([0.1, 17])
axs[0].set_ylabel('$\eta$ [bits/s/Hz]')
axs[0].legend(['Shannon limit', 'Adaptive system (BER < $10^{-6}$)', 'Adaptive system (BER < $10^{-3}$)'], fontsize=8)
for ax in axs.flat:
    ax.label_outer()
    ax.set_xlim([-3, 35])
axs[0].set_yticks([0.1, 1, 10], ['0.1', '1', '10'])
axs[1].plot(eb_n0_db, 100 * ratio6, color='darkred')
axs[1].plot(eb_n0_db, 100 * ratio3, color='darkred', linestyle='--', alpha=0.4)
axs[1].set_ylim([-3, 103])
axs[1].set_xlabel('$E_b/N_0$')
axs[1].set_ylabel("Utilization of cap. [%]")
axs[1].set_yticks([0, 25, 50, 75, 100], [0, 25, 50, 75, 100])
plt.tight_layout()
axs[1].grid(alpha=0.4)
axs[0].grid(alpha=0.4, which='both')
plt.savefig("plots/shannon_limit.pdf")
plt.show()
