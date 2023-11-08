import numpy as np
import matplotlib.pyplot as plt

from gwpy.timeseries import TimeSeries
from scipy.signal.windows import tukey

html_dir = "/home/urash/twouters/public_html/GW_data_analysis/GW170817"

params = {"axes.grid": True,
          "text.usetex" : True,
          "font.family" : "serif",
          "ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
          "font.serif" : ["Computer Modern Serif"],
          "xtick.labelsize": 16,
          "ytick.labelsize": 16,
          "axes.labelsize": 16,
          "legend.fontsize": 16,
          "legend.title_fontsize": 16,
          "figure.titlesize": 16}

plt.rcParams.update(params)


# Location of PSD data peter
plt.figure(figsize=(12, 8))
for file, label in zip(['h1_psd.txt', 'l1_psd.txt'], ["H1", "L1"]):
    psd_dir = "/home/urash/twouters/GW_data/GW170817_data"

    psd_file = np.loadtxt(f"{psd_dir}/{file}")

    f   = psd_file[:, 0]
    psd = psd_file[:, 1]

    plt.plot(f, psd, label = label, zorder = 1000)

plt.legend()
plt.yscale('log')
plt.xlabel('Frequencies [Hz]')
plt.ylabel('PSD')
plt.savefig(f"{html_dir}/figures/bayeswave_psd.png", bbox_inches = 'tight')
plt.close()