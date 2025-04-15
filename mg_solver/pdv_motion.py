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
import cpp_modules._planck_integral as planck_integral

"""
Solve the multigroup radiation-matter equations under prescribed homologous motion
Radiation and Material PdV terms are included
The Doppler term of the radiation field is included

An analytic solution exists for:
* No material-radiation coupling (No absorption and Compton scattering)
"""

rho0 = 1.

# ---- constant velocity homologous motion
compression = True
# compression = False
tau_h = 1e-9
if compression: tau_h *= -1.
alpha = lambda t: 1.+t/tau_h
rho = lambda t: rho0/alpha(t)**3
div_velocity = lambda t: 3./(tau_h*alpha(t))

t_end = tau_h*(8.**((-1. if compression else 1.)*1./3. )-1.)


# # case 1
# T_mat_init = 20.*units.kev_kelvin
# T_rad_init = 1.*units.kev_kelvin
# epsilon = 0.7

# # # case 2
T_mat_init = 1.*units.kev_kelvin
T_rad_init = 1.*units.kev_kelvin
epsilon = 0.1

# ---- material EOS - u_mat~T^4
T_mat_u_eos = lambda um,rho: (um*epsilon/units.arad)**0.25
u_mat_T_eos = lambda T,rho: units.arad/epsilon*T**4
gamma_mat = 5./3.

# ---- energy groups
G = 100
# G = 50
# G = 500
Tmax = max(T_mat_init, T_rad_init)
Tmin = min(T_mat_init, T_rad_init)
E_high = 20.*units.k_boltz*Tmax
E_low = 1e-3*units.k_boltz*Tmin

energy_groups_boundaries = np.array(sorted(list(set(
    list(np.linspace(E_low, E_high, G+1))+\
    # list(np.geomspace(E_low, E_high, G+1)) +\
    [1.*units.k_boltz*T_rad_init, 6.*units.k_boltz*T_rad_init]

))))
energy_groups_width = np.diff(energy_groups_boundaries)
energy_groups_centers = 0.5*(energy_groups_boundaries[1:] + energy_groups_boundaries[:-1])
G = len(energy_groups_centers)

# initial radiation field
Eg_init = get_planckian(T_rad_init, energy_groups_boundaries)
Eg_init[energy_groups_centers/(units.k_boltz*T_rad_init) >= 6.] = 0.
Eg_init[energy_groups_centers/(units.k_boltz*T_rad_init) <= 1.] = 0.


normalized_planck = lambda x: 15.*x*x*x*np.exp(-x)/(np.pi*np.pi*np.pi*np.pi*(1-np.exp(-x)))

# # plot initial spec
# plt.stairs(edges=energy_groups_boundaries/units.kev, values=Eg_init/energy_groups_width)
# eee = np.linspace(energy_groups_boundaries[0]*1.001,energy_groups_boundaries[-1]*0.99,1000)
# planck_eee = normalized_planck(eee/(units.k_boltz*T_rad_init))*units.arad*T_rad_init**4/(units.k_boltz*T_rad_init)
# plt.plot(eee/units.kev, planck_eee, "-.k")
# plt.show()

# ----- material opacity - must be constant to have analytic solution
# sigma0 = 1.5
sigma0 = 0.1
sigma_absorption = lambda T,rho: [sigma0]*G

solver = MGSolver(
    energy_groups_boundaries=energy_groups_boundaries,
    energy_groups_centers=energy_groups_centers,
    tau_compton=None,
    sigma_absorption=None,
    # sigma_absorption=sigma_absorption,
    u_mat_T_eos=u_mat_T_eos,
    T_mat_u_eos=T_mat_u_eos,
    Eg_init=Eg_init,
    T_mat_init=T_mat_init, 
    rho=rho,        
    div_velocity=div_velocity,
    gamma_mat=gamma_mat,
    # doppler_term=False,
    doppler_term=True,
)

times = np.array(sorted(list(set(
    list(np.geomspace(t_end*1e-10, t_end, 20))+\
    list(np.linspace(t_end*1e-10, t_end, 100))
))))

res = [solver.solve(time=t) for t in times]
T_mat = np.array([r["T_mat"]/units.kev_kelvin for r in res])
T_rad = np.array([r["T_rad"]/units.kev_kelvin for r in res])
E_rad = np.array([r["E_rad_tot"] for r in res])
u_mat = np.array([r["u_mat"] for r in res])
rho_rho0 = rho(times)/rho(0.)
E_rad_0 = np.sum(Eg_init)
u_mat_0 = u_mat_T_eos(T_mat_init, rho(0.))

from tabulate import tabulate
table = tabulate(list(zip(times[::-5][::-1]/1e-9, T_mat[::-5][::-1], T_rad[::-5][::-1])), headers=["time [ns]", "T_mat [kev]", "T_rad [kev]"], floatfmt=".10g", numalign="left")
print(table)


no_coupling = solver.sigma_absorption==None and solver.tau_compton==None 
plt.plot(rho_rho0, E_rad/E_rad_0, "b", lw=2., label="$E_{{rad}}(t)/E_{{rad}}(t=0)$")
if no_coupling: plt.plot(rho_rho0, rho_rho0**(4./3.), "--r", lw=2., label="$y=x^{{4/3}}$")
plt.plot(rho_rho0, u_mat/u_mat_0, "k", lw=2., label="$u_{{m}}(t)/u_{{m}}(t=0)$")
if no_coupling: plt.plot(rho_rho0, rho_rho0**(gamma_mat), c="lime", ls="--", lw=2., label="$y=x^{{\\gamma_{{m}}}}$")
plt.grid()
plt.legend(fontsize=18, loc="best").set_draggable(True)
plt.xlabel("$\\rho(t)/\\rho(t=0)$")
plt.ylabel("energy density ratio")
# plt.xscale("log")
plt.tight_layout()
plt.savefig("gamma_law_scaling.png")
plt.show()


plt.plot(times/1e-9, T_mat, "r", lw=2, label=f"$T_{{m}}(t={times[-1]/1e-9:.3g}\\mathrm{{ns}})={T_mat[-1]:g}$")
plt.plot(times/1e-9, T_rad, "b", lw=2, label=f"$T_{{r}}(t={times[-1]/1e-9:.3g}\\mathrm{{ns}})={T_rad[-1]:g}$")
plt.grid()
plt.legend(fontsize=18, loc="best").set_draggable(True)
plt.ylabel("$T(t)$ [kev]")
plt.xlabel("$t$ [ns]")
plt.xscale("log")
plt.xlim(xmin=1e-5*t_end/1e-9)
plt.tight_layout()
plt.savefig("pdv_temps.png")
plt.show()

# --- plot spectra at different times
# to make a movie run `ml ffmpeg/7.1; ml x264; ffmpeg -framerate 3 -pattern_type glob -i 'spec_loglog_*.png' -c:v libx264 -pix_fmt yuv420p out.mp4`
dir_figs = "spec_output"
import os 
os.makedirs(dir_figs, exist_ok=True)    
for i, (t,r) in enumerate(zip(times[::-5][::-1], res[::-5][::-1])):
    Tm = r["T_mat"]
    Tr = r["T_rad"]
    Eg = r["Eg"]
    
    # initial spectra
    spec_init = Eg_init / (energy_groups_width/units.kev)
    plt.stairs(edges=energy_groups_boundaries/units.kev, values=1e-13*spec_init, color="k", linestyle="-", linewidth=2., label="$t=0$")
    
    spec = Eg / (energy_groups_width/units.kev)
    plt.stairs(edges=energy_groups_boundaries/units.kev, values=1e-13*spec, color="b", linestyle="-", linewidth=2., label="Numerical")
    
    # analytic solution with Doppler term
    ec_plot = np.linspace(energy_groups_boundaries[0]*1.001,energy_groups_boundaries[-1]*0.999,1000)
    kTr_init = units.k_boltz*T_rad_init
    initial_spec_func = np.vectorize(lambda e: units.arad*T_rad_init**4*normalized_planck(e/kTr_init)/kTr_init if 1.<e/kTr_init<6. else 0.)
    rho_rho0_i = rho(t)/rho(0.)
    plt.plot(ec_plot/units.kev, 1e-13*rho_rho0_i*initial_spec_func(ec_plot/rho_rho0_i**(1./3.))*units.kev, "--r", lw=2., label="exact Doppler")
    
    # analytic solution without Doppler term
    plt.stairs(edges=energy_groups_boundaries/units.kev, values=1e-13*rho_rho0_i**(4./3.)*spec_init, color="orange", linestyle="-.", linewidth=2., label="exact no-Doppler")
    

    # Planckian at the current material temperature
    plt.plot(energy_groups_centers/units.kev, 1e-13*normalized_planck(energy_groups_centers/(units.k_boltz*Tm))*units.arad*Tm**4/(units.k_boltz*Tm/units.kev), "--g", lw=2., label=f"$U_{{P}}(T_{{m}}={Tm/units.kev_kelvin:g}\\mathrm{{kev}})$")
    
    # Planckian at the current effective radiation temperature
    plt.plot(energy_groups_centers/units.kev, 1e-13*normalized_planck(energy_groups_centers/(units.k_boltz*Tr))*units.arad*Tr**4/(units.k_boltz*Tr/units.kev), ls="-.", c="lime", lw=2., label=f"$U_{{P}}(T_{{r}}={Tr/units.kev_kelvin:g}\\mathrm{{kev}})$")

    plt.legend(loc="best", fontsize=16)
    plt.xlim([energy_groups_boundaries[0]/units.kev, energy_groups_boundaries[-1]/units.kev])
    plt.grid()
    plt.title(f"$t={t/1e-9:.4g}\\mathrm{{ns}}$ $\\rho(t)/\\rho(0)={rho_rho0_i:.4g}$")
    plt.xlabel("photon energy $E$ [kev]")
    plt.ylabel("$U(E) \\ [10^{{13}}\\mathrm{{erg/cm^{{3}}/kev}}]$\n")
    plt.tight_layout()
    plt.savefig(path.join(dir_figs, f"spec_{i:04d}.png"))
    # plt.show()
    plt.ylim(ymin=1e-2)
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig(path.join(dir_figs, f"spec_logx_{i:04d}.png"))
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(path.join(dir_figs, f"spec_loglog_{i:04d}.png"))
    plt.close()
    print(f"spec {i}")