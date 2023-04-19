import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from uncertainties import ufloat
from uncertainties import unumpy as unp

# 2 Determination of the average decay length of the K+ ###########################################
# Functions -------------------------------------
def P(x, l_p):
    return 1/l_p * np.exp(- 1/l_p * x)

def K(x, l_k):
    return 1/l_k * np.exp(- 1/l_k * x)

def T(x, l_k):
    l_p = 4188 # m
    return 0.84 * P(x,l_p) + 0.16 * K(x,l_k)

def likelihood_of_data(data, l_k):
    '''
        returns the log of the likelihood because the likelihood gets lower then the computer precision.
    '''
    log_of_likelihood = 0
    for x in data:
        log_of_likelihood += np.log(T(x, l_k))
    return log_of_likelihood

def function_to_minimize(variables, data):
    '''
        We want to maximize the likelihood, but can only minimize a function.
        => minimize -likelihood
    '''
    l_k = variables[0]
    return -likelihood_of_data(data, l_k)

# load data -------------------------------------
data = np.loadtxt('dec_lengths.txt')

# fit -------------------------------------------
res = minimize(function_to_minimize, x0=[500], args=(data), method='Powell')
mean_decay_lenght_Kaon = res.x[0]
print(f'mean decay lenght Kaon {mean_decay_lenght_Kaon} m')

# uncertainties ---------------------------------
mean_decay_lenght_Kaon_uncertainty = 0 # TODO

# plot ------------------------------------------
counts, bins, _ = plt.hist(data, bins=500, density=True, label='Data')
xdata = bins[:-1]
plt.plot(xdata,T(xdata, res.x[0]), label='Fit')
plt.xlabel('Decay Length [m]')
plt.ylabel('Number of Particles (normalized)')
plt.title(r'Particle decay length of mixed $K^+$ and $\pi^+$ beam')
plt.legend()
plt.savefig('decay_length_of_mixed_beam.pdf')
plt.show()

# mean life time of Kaon ------------------------
l_k = ufloat(mean_decay_lenght_Kaon,mean_decay_lenght_Kaon_uncertainty) # m
l_p = ufloat(4188,0) # m
m_k = ufloat(493.677, 0.016) # MeV
m_p = ufloat(139.57039, 0.00018) # MeV
tau_p = ufloat(2.6033e-8, 0.0005e-8) # s

tau_k = l_k/l_p * tau_p * m_k/m_p
print(r'measured mean life time $\tau_{K^+} =', '{:L}$'.format(tau_k), 's')
print(r'literature mean life time $\tau_{K^+} = \left(1.2380 \pm 0.0020\right) \times 10^{-8}$ s')

# 3 Infinitely narrow beam along the z axis #######################################################




# 4 Divergent beam ################################################################################




