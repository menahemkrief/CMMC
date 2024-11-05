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
from cpp_modules._compton_matrix_mc import ComptonMatrixMC
import cpp_modules._units as units

"""
Solution for two Benchmarks in the literature:
Figs 1-2 in Winslow AM. Multifrequency-gray method for radiation diffusion with Compton scattering. Journal of Computational Physics. 1995 Mar 15;117(2):262-73.
Figs 1-2 in McGraw C, Till A, Warsa J. A new operator-split Compton scattering method. Journal of Computational Physics. 2023 Apr 1;478:111980.
"""

rho = 1.

# # case 1 (Figs. 1-2 in Winslaw 1995 and  Fig. 1 in McGraw-Till-Warsa 2023)
# T_mat_init = 20.*units.kev_kelvin
# T_rad_init = 1.*units.kev_kelvin
# t_end = 3e-8 # McGraw-Till-Warsa paper
# # t_end = 5e-9 # Winslaw1995 paper

# case 2 (Fig. 2 in McGraw-Till-Warsa 2023)
T_mat_init = 1.*units.kev_kelvin
T_rad_init = 10.*units.kev_kelvin
t_end = 0.5e-8


# ---- material EOS
Zf = 1.
A = 1.
gamma = 5./3.
cv = (Zf+1.)*units.k_boltz*units.Navogadro/(gamma-1.)/A # erg/g/K

T_mat_u_eos = lambda um,rho: um/(rho*cv)
u_mat_T_eos = lambda T,rho: T*(rho*cv)

# ---- energy groups
E_high = 200.*units.kev
E_low = 1e-3*units.kev

G = 32
# # G =64
energy_groups_boundaries = np.array(sorted(list(set(
    # list(np.linspace(E_low, E_high, 33))+\
    list(np.geomspace(E_low, E_high, G+1))+\
    [400.*units.kev]
))))
energy_groups_centers = 0.5*(energy_groups_boundaries[1:] + energy_groups_boundaries[:-1])
G = len(energy_groups_centers)

# initial radiation field
Eg_init = get_planckian(T_rad_init, energy_groups_boundaries)

# ----- material opacity
# sigma_absorption = None
# # Free-Free opacity of fully ionized Hydrogen (Rybicky & Lightman book, Eq. 5.18b)
sigma_ff_RL = lambda e,T,rho: 3.7e8*Zf**3*(rho*units.Navogadro/A)**2*T**(-0.5)*(1.-np.exp(-e/(units.k_boltz*T)))*(e/units.planck_constant)**-3
energy_groups_centers_geom = np.sqrt(energy_groups_boundaries[1:]*energy_groups_boundaries[:-1])
sigma_absorption = lambda T,rho: [sigma_ff_RL(max(e, 10.*units.ev),T,rho) for e in energy_groups_centers_geom]

# --plot the opacity on bins
# plt.plot(energy_groups_centers/units.kev, sigma_absorption2(2.*units.kev_kelvin,1.))
# plt.plot(energy_groups_centers/units.kev, sigma_absorption(2.*units.kev_kelvin,1.))
# plt.stairs(edges=energy_groups_boundaries/units.kev, values=sigma_absorption(5.*units.kev_kelvin,1.) ,label=f"R&L")
# plt.xscale("log")
# plt.yscale("log")
# plt.show()

# # ---- compton matrix
# tau_compton = None
# # # tau_compton = lambda T,rho: np.ones((G,G))*0.01

compton_engine = ComptonMatrixMC(
    energy_groups_centers=energy_groups_centers,
    energy_groups_boundaries=energy_groups_boundaries, 
    num_of_samples=int(2e5),
    force_detailed_balance=True,
    seed=0,
)
compton_engine.set_tables(temperature_grid=np.linspace(min(T_mat_init, T_rad_init)*0.95, max(T_mat_init, T_rad_init), 10))
tau_compton = lambda T,rho: np.asfarray(compton_engine.get_tau_matrix(temperature=T, density=rho, A=A, Z=Zf))

# # --- Multigroup radiation-material equations solver
solver = MGSolver(
    energy_groups_boundaries=energy_groups_boundaries,
    energy_groups_centers=energy_groups_centers,
    sigma_absorption=sigma_absorption,
    tau_compton=tau_compton,
    induced_scattering=True,
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
res0 = dict(T_mat=T_mat_init, T_rad=T_rad_init, Eg=Eg_init, u_mat=u_mat_T_eos(T_mat_init,rho))
res =  [res0] + res
times = np.array([0.]+list(times))

T_mat = np.array([r["T_mat"]/units.kev_kelvin for r in res])
T_rad = np.array([r["T_rad"]/units.kev_kelvin for r in res])

from tabulate import tabulate
table = tabulate(list(zip(times[::5]/1e-9, T_mat[::5], T_rad[::5])), headers=["time [ns]", "T_mat [kev]", "T_rad [kev]"], floatfmt=".10g", numalign="left")

# ---- plot material and effective radiation temperatures as a function of time
plt.plot(times/1e-9, T_mat, "r", lw=2, label=f"$T_{{m}}(t={times[-1]/1e-9:.3g}\\mathrm{{ns}})={T_mat[-1]:g}$")
plt.plot(times/1e-9, T_rad, "b", lw=2, label=f"$T_{{r}}(t={times[-1]/1e-9:.3g}\\mathrm{{ns}})={T_rad[-1]:g}$")
plt.axhline(y=solver.T_eq/units.kev_kelvin, c="k", ls="--", lw=3, label=f"$T_{{eq}}={solver.T_eq/units.kev_kelvin:g}$")
plt.grid()
plt.ylabel("$T(t)$ [kev]")
plt.xlabel("$t$ [ns]")
plt.tight_layout()

# ---- compare to data from literature

if  T_mat_init == 20.*units.kev_kelvin:

    if sigma_absorption != None:
        # First case in McGraw-Till-Warsa (Fig. 1)
        data = np.loadtxt("data_from_literature/Till_winslow_Tmat.txt", delimiter=",")
        plt.plot(data[:,0]/1e-9, data[:,1], "ro",  label=f"$T_{{m}}$ McGraw-Till-Warsa")
        data = np.loadtxt("data_from_literature/Till_winslow_Trad.txt", delimiter=",")
        plt.plot(data[:,0]/1e-9, data[:,1], "bo", label=f"$T_{{r}}$ McGraw-Till-Warsa")
        data = np.loadtxt("data_from_literature/Winslow_Tmat.txt", delimiter=",")
        plt.plot(data[:,0]/1e-9, data[:,1], "rs",  label=f"$T_{{m}}$ Winslow")
        data = np.loadtxt("data_from_literature/Winslow_Trad.txt", delimiter=",")
        plt.plot(data[:,0]/1e-9, data[:,1], "bs", label=f"$T_{{r}}$ Winslow")
        plt.legend(loc="best", fontsize=15).set_draggable(True)
        plt.savefig("WinslowTill.png")
        plt.savefig("WinslowTill.pdf")
        # as in Winslow1995 Fig. 1
        plt.xlim([0,5])
        plt.savefig("WinslowTill_xlim5.png")
        plt.savefig("WinslowTill_xlim5.pdf")
    else:
        # Winslow1995 Fig. 2
        data = np.loadtxt("data_from_literature/Winslow_Tmat_only_compton.txt", delimiter=",")
        plt.plot(data[:,0]/1e-9, data[:,1], "rs",  label=f"$T_{{m}}$ Winslow")
        data = np.loadtxt("data_from_literature/Winslow_Trad_only_compton.txt", delimiter=",")
        plt.plot(data[:,0]/1e-9, data[:,1], "bs", label=f"$T_{{r}}$ Winslow")
        plt.legend(loc="best").set_draggable(True)
        plt.xlim([0,5])
        plt.savefig("Winslow_only_compton.png")
        plt.savefig("Winslow_only_compton.pdf")
else:
    # second case in McGraw-Till-Warsa (Fig. 2)
    data = np.loadtxt("data_from_literature/Till_Tmat.txt", delimiter=",")
    plt.plot(data[:,0]/1e-9, data[:,1], "ro",  label=f"$T_{{m}}$ McGraw-Till-Warsa")
    plt.legend(loc="best").set_draggable(True)
    plt.xscale("log")
    plt.xlim([1e-2,5])
    plt.savefig("Till_Fig2.png")
    plt.savefig("Till_Fig2.pdf")

plt.show()

# --- plot spectra at different times
# to make a movie run `ml ffmpeg/7.1; ml x264; ffmpeg -framerate 3 -pattern_type glob -i 'spec_loglog_*.png' -c:v libx264 -pix_fmt yuv420p out.mp4`
normalized_planck = lambda x: 15.*x*x*x*np.exp(-x)/(np.pi*np.pi*np.pi*np.pi*(1-np.exp(-x)))
energy_groups_width = np.diff(energy_groups_boundaries)
dir_figs = "spec_output"
import os 
os.makedirs(dir_figs, exist_ok=True)    
for i, (t,r) in enumerate(zip(times[::5], res[::5])):
    Tm = r["T_mat"]
    Tr = r["T_rad"]
    Eg = r["Eg"]
    spec = Eg / (energy_groups_width/units.kev)
    plt.stairs(edges=energy_groups_boundaries/units.kev, values=spec, color="b", linestyle="-", linewidth=2., label="Numerical")
    
    # Planckian at the current material temperature
    plt.plot(energy_groups_centers/units.kev, normalized_planck(energy_groups_centers/(units.k_boltz*Tm))*units.arad*Tm**4/(units.k_boltz*Tm/units.kev), "--g", lw=2., label=f"$U_{{P}}(T_{{m}}={Tm/units.kev_kelvin:g}\\mathrm{{kev}})$")
    
    # Planckian at the current effective radiation temperature
    plt.plot(energy_groups_centers/units.kev, normalized_planck(energy_groups_centers/(units.k_boltz*Tr))*units.arad*Tr**4/(units.k_boltz*Tr/units.kev), "-.r", lw=2., label=f"$U_{{P}}(T_{{r}}={Tr/units.kev_kelvin:g}\\mathrm{{kev}})$")

    plt.legend(loc="best", fontsize=16)
    plt.xlim([energy_groups_boundaries[0]/units.kev, energy_groups_boundaries[-1]/units.kev])
    plt.grid()
    plt.title(f"$t={t/1e-9:g}$ns")
    plt.xlabel("photon energy $E$ [kev]")
    plt.ylabel("$U(E) \\ [\\mathrm{{erg/cm^{{3}}/kev}}]$ ")
    plt.tight_layout()
    plt.savefig(path.join(dir_figs, f"spec_{i:04d}.png"))
    plt.ylim(ymin=1e-2)
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig(path.join(dir_figs, f"spec_logx_{i:04d}.png"))
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(path.join(dir_figs, f"spec_loglog_{i:04d}.png"))
    # plt.show()
    plt.close()
    print(f"spec {i}")