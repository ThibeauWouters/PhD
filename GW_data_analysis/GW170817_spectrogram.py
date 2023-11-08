
### NOTE
# This requires Jim and hence has to be ran on the Potsdam machines
###

import time
from jimgw.jim import Jim
from jimgw.detector import H1, L1
from jimgw.likelihood import HeterodynedTransientLikelihoodFD, TransientLikelihoodFD
# from jimgw.waveform import RippleIMRPhenomD
# from jimgw.waveform import RippleTaylorF2
from jimgw.prior import Uniform
import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)

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

def get_data(name, trigger_time, gps_start_pad, gps_end_pad, tukey_alpha=0.2):
    data_td = TimeSeries.fetch_open_data(name, trigger_time - gps_start_pad, trigger_time + gps_end_pad, cache=True)
    segment_length = data_td.duration.value
    n = len(data_td)
    delta_t = data_td.dt.value
    # data = jnp.fft.rfft(jnp.array(data_td.value)*tukey(n, tukey_alpha))*delta_t
    # freq = jnp.fft.rfftfreq(n, delta_t)
    
    # data = data_td.value * tukey(n, tukey_alpha) * delta_t
    
    return data_td


### Get the data that Jim fetches as well
gps = 1187008882.43

T = 2 * 128
start = gps - T/2
end   = gps + T/2
fmin = 20.0
fmax = 2048.0

ifos = ["H1", "L1"]

# H1.load_data(gps, T/2, T/2, fmin, fmax, psd_pad=16, tukey_alpha=0.2)
# L1.load_data(gps, T/2, T/2, fmin, fmax, psd_pad=16, tukey_alpha=0.2)

H1_data = get_data("H1", gps, T/2, T/2)
L1_data = get_data("L1", gps, T/2, T/2)


# Create spectrogram from Jim data
stride = 20
specgram_H1 = H1_data.spectrogram(stride, fftlength=8, overlap=4) ** (1/2.)
specgram_L1 = L1_data.spectrogram(stride, fftlength=8, overlap=4) ** (1/2.)

# Create plot
plt.subplots(2, 1, figsize = (15, 6))

plt.subplot(2, 1, 1)
# plt.plot(specgram_H1.times, specgram_H1.frequencies)
specgram_H1.plot(norm='log', vmin=1e-23, vmax=1e-19)
plt.title("Hanford")
plt.subplot(2, 1, 2)
# plt.plot(specgram_L1.times, specgram_L1.frequencies)
specgram_L1.plot(norm='log', vmin=1e-23, vmax=1e-19)
plt.title("Livingston")

plt.title("Data fetched by jim")
plt.savefig(f"{html_dir}/figures/spectrogram_jim.png", bbox_inches='tight')
plt.close()

### Load in Peter's data



# ### Plot Peter's data
# plt.plot(H1.times, H1.value)
# plt.plot(L1.times, L1.value)

# # plt.title("Data fetched by jim")
# # plt.savefig(f"{html_dir}/figures/strain_peter.png", bbox_inches='tight')
# plt.close()

