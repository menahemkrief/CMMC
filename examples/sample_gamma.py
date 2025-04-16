import numpy as np
from matplotlib import pyplot as plt
from scipy.special import kn as Bessel_Kn

import sys
# add the cpp modules to the system's path
from os import path
sys.path.append(path.abspath(path.join(path.dirname(path.abspath(__file__)), path.pardir)))
from cpp_modules._compton_matrix_mc import ComptonMatrixMC
import cpp_modules._units as units


def P(x_P, theta_P):
    Db = 2*theta_P**3*np.exp(-1.0/theta_P)*(1.0 + 1.0/theta_P + 0.5/theta_P**2.0)
    return x_P**2*np.exp(-x_P/theta_P)/Db

def MJ(x_MJ, theta_MJ):
    Db_MJ = theta_MJ * Bessel_Kn(2, 1.0/theta_MJ)
    beta_MJ = np.sqrt(1.0 - 1.0/x_MJ**2)
    return x_MJ**2*beta_MJ*np.exp(-x_MJ/theta_MJ)/Db_MJ

T = 2.0*units.me_c2 / units.k_boltz
theta = units.k_boltz * T / units.me_c2

tau_generator = ComptonMatrixMC(compton_temperatures=[1.0, 2.0], energy_groups_centers=[0.1], energy_groups_boundaries=[0.0, 0.2], num_of_samples=10000)

x = np.linspace(1.0, 20.0, 1000)
y = P(x_P=x, theta_P=theta)

n = 100000
samples = np.zeros(n)
for i in range(n):
    samples[i] = tau_generator.sample_gamma(temperature=T)

plt.plot(x, y)
plt.hist(samples, bins = n//500, density=True)

plt.xlim(1.0)
plt.grid()
plt.show()


x = np.geomspace(1.0, 100.0, 1000)
y = MJ(x_MJ=x, theta_MJ=theta)
# y = MJ(x=x, theta=0.1)

n = 100000
samples = np.zeros(n)
weights = np.zeros(n)
for i in range(n):
    gamma_sample = tau_generator.sample_gamma(temperature=T)
    weights[i] = np.sqrt(1.0 - 1.0 / gamma_sample**2.0) 
    samples[i] = gamma_sample

plt.plot(x, y)
# plt.plot(x, MJ(x_MJ=x, theta_MJ=0.1))
# plt.plot(x, MJ(x_MJ=x, theta_MJ=1.0))
plt.hist(samples, bins = n//100, weights=weights, density=True)
plt.ylim(1e-10, 1e1)
plt.xscale("log")
plt.yscale("log")
plt.xlim(1.0)
plt.grid()
plt.show()