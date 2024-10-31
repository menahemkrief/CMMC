import numpy as np
import sys

from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('classic')
matplotlib.rcParams.update({
    'font.size': 17,        # Default font size
})

from mg_solver import MGSolver, get_planckian

# add the cpp modules to the system's path
from os import path
sys.path.append(path.abspath(path.join(path.dirname(path.abspath(__file__)), path.pardir)))
import cpp_modules._units as units

"""
An analytic example for the solution of the multigroup radiation-matter equations
The solution exists for:
* No Compton scattering
* A constant opacity [does not depend on photon energy or material temperature]
* A material energy density of the form u(T)=aT^4/epsilon
"""
rho = 1.

# # # case 1
# T_mat_init = 20.*units.kev_kelvin
# T_rad_init = 1.*units.kev_kelvin
# t_end = 3e-9
# epsilon = 0.7

# # case 2
T_mat_init = 1.*units.kev_kelvin
T_rad_init = 10.*units.kev_kelvin
t_end = 0.5e-9
epsilon = 0.1


# ---- material EOS - u_mat~T^4 
T_mat_u_eos = lambda um,rho: (um*epsilon/units.arad)**0.25
u_mat_T_eos = lambda T,rho: units.arad/epsilon*T**4

# ---- energy groups
G = 10
Tmax = max(T_mat_init, T_rad_init)
Tmin = min(T_mat_init, T_rad_init)
E_high = 20.*units.k_boltz*Tmax
E_low = 1e-3*units.k_boltz*Tmin

energy_groups_boundaries = np.array(sorted(list(set(
    # list(np.linspace(E_low, E_high, 10))+\
    list(np.geomspace(E_low, E_high, G+1))
))))
energy_groups_centers = 0.5*(energy_groups_boundaries[1:] + energy_groups_boundaries[:-1])

Eg_init = get_planckian(T_rad_init, energy_groups_boundaries)
G = len(Eg_init)

# ----- material opacity - must be constant to have analytic solution
sigma0 = 1.5
sigma_absorption = lambda T,rho: [sigma0]*G

solver = MGSolver(
    energy_groups_boundaries=energy_groups_boundaries,
    energy_groups_centers=energy_groups_centers,
    sigma_absorption=sigma_absorption,
    tau_compton=None,
    u_mat_T_eos=u_mat_T_eos,
    T_mat_u_eos=T_mat_u_eos,
    Eg_init=Eg_init,
    T_mat_init=T_mat_init, 
    rho=rho,        
)

times = np.array(sorted(list(set(
    list(np.geomspace(t_end*1e-10, t_end, 100))+\
    list(np.linspace(t_end*1e-10, t_end, 100))
))))

res = [solver.solve(time=t) for t in times]
T_mat = np.array([r["T_mat"]/units.kev_kelvin for r in res])
T_rad = np.array([r["T_rad"]/units.kev_kelvin for r in res])

from tabulate import tabulate
table = tabulate(list(zip(times[::5]/1e-9, T_mat[::5], T_rad[::5])), headers=["time [ns]", "T_mat [kev]", "T_rad [kev]"], floatfmt=".10g", numalign="left")
print(table)

plt.plot(times/1e-9, T_mat, "r", lw=2, label=f"$T_{{m}}(t={times[-1]/1e-9:.3g}\\mathrm{{ns}})={T_mat[-1]:g}$")
plt.plot(times/1e-9, T_rad, "b", lw=2, label=f"$T_{{r}}(t={times[-1]/1e-9:.3g}\\mathrm{{ns}})={T_rad[-1]:g}$")

# analytic solution
E0 = units.arad*T_rad_init**4
U0 = units.arad*T_mat_init**4
expp = np.exp(-(1.+epsilon)*units.clight*sigma0*times)
Erad_anal = ((E0-U0)*expp+(U0+epsilon*E0))/(1.+epsilon)
T_rad_anal = (Erad_anal/units.arad)**0.25 / units.kev_kelvin
Umat_anal = (epsilon*(U0-E0)*expp+(U0+epsilon*E0))/(1.+epsilon)
T_mat_anal = (Umat_anal/units.arad)**0.25 / units.kev_kelvin
plt.plot(times/1e-9, T_rad_anal, c="fuchsia", ls="--", lw=2, label=f"Anal $T_{{r}}(t={times[-1]/1e-9:.3g}\\mathrm{{ns}})={T_mat[-1]:g}$")
plt.plot(times/1e-9, T_mat_anal, c="lime",    ls="--", lw=2, label=f"Anal $T_{{m}}(t={times[-1]/1e-9:.3g}\\mathrm{{ns}})={T_mat[-1]:g}$")

plt.axhline(y=solver.T_eq/units.kev_kelvin, c="k", ls="--", lw=3, label=f"$T_{{eq}}={solver.T_eq/units.kev_kelvin:g}$")
plt.grid()
plt.legend(fontsize=15, loc="best").set_draggable(True)
plt.ylabel("$T(t)$ [kev]")
plt.xlabel("$t$ [ns]")
plt.xscale("log")
plt.xlim(xmin=1e-5*t_end/1e-9)
plt.show()