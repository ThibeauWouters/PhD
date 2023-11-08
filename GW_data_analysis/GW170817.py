# Quick script to analyze the results of GW170817 parameters

import numpy as np

def Mc_q_to_ms(Mchirp, q):
    """
    Converts chirp mass and symmetric mass ratio to binary component masses.
    """
    
    m2 = 1/(1+q) * (q/(1+q)**2) ** (-3/5) * Mchirp
    m1 = q * m2
    
    return m1, m2

def ms_to_Mc_eta(m):
    """
    Converts binary component masses to chirp mass and symmetric mass ratio.
    """
    m1, m2 = m
    Mchirp = (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)
    q = m1/m2
    return Mchirp, q

def effective_spin(m1, chi1, m2, chi2):
    return (m1 * chi1 + m1 * chi1) / (m1 +m2)

def lambda_tilde(m1, lambda1, m2, lambda2):
    
    return 16./13. * ((m1 + 12 * m2) * m1 ** 4 * lambda1 + (m2 + 12 * m1) * m2 ** 4 * lambda2) / ((m1 + m2) ** 5)

Mc = 1.228
q = 1 / 0.871

m1, m2 = Mc_q_to_ms(Mc, q)

chi1 = 0.033
chi2 = 0.029

lambda1 = 3131
lambda2 = 2147

eff_chi = effective_spin(m1, m2, chi1, chi2)
lamb = lambda_tilde(m1, lambda1, chi1, lambda2)

nb_digits_round = 3
print(f"eff_chi: {np.round(eff_chi, nb_digits_round)}")
print(f"lambda tilde: {np.round(lamb, nb_digits_round)}")


### Load GWF file

from gwpy.timeseries import TimeSeries

my_directory = "/home/urash/twouters/GW_data/GW170817_data"
my_file = "/H-H1_LOSC_CLN_16_V1-1187007040-2048.gwf"

data = TimeSeries.read(my_directory + my_file, "H1:LOSC-STRAIN", format="gwf")
print(data)