import numpy as np

from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('classic')
matplotlib.rcParams.update({
    'font.size': 17,        # Default font size
})

import sys
# add the cpp modules to the system's path
from os import path
import os
sys.path.append(path.abspath(path.join(path.dirname(path.abspath(__file__)), path.pardir)))
from cpp_modules._compton_matrix_mc import ComptonMatrixMC
import cpp_modules._units as units

"""
Compares the multigroup compton matrices to the pointwise results from
the Figs. in Pomraning book pages 188-189
"""

for i, case in enumerate([
    dict(T=1., emax=75.,   ein=[5.,10.,20.,40.,60.], ylim=[1,1e4],    name="Pomeraning_1kev_low"),
    dict(T=1., emax=340.,  ein=[80.,120.,200.,300.], ylim=[1e-2,1e2], name="Pomeraning_1kev_high"),
    # dict(T=20., emax=25., ein=[5.,10.], ylim=[1e-1,1e3], name="Pomeraning_20kev_low"),
    dict(T=20., emax=140., ein=[5.,10.,20.,40.,60.], ylim=[1e-1,1e3], name="Pomeraning_20kev_low"),
    dict(T=20., emax=440., ein=[80.,120.,200.,300.], ylim=[1e-2,1e2], name="Pomeraning_20kev_high"),
    ]):
    
    barn = 1e-24
    mbarn = 1e-3*barn

    eb = np.array(sorted(list(set(
        list(np.linspace(0.01, case["emax"]/10, 50))+\
        list(np.linspace(0.01, case["emax"], 100*4))+\
        list(np.geomspace(0.01, case["emax"], 50))
    ))))
    
    eb *= units.kev
    ec = 0.5*(eb[1:] + eb[:-1])
    ewid = np.diff(eb)
    T = case["T"]*units.kev_kelvin
    compton_engine = ComptonMatrixMC(
        energy_groups_centers=ec, energy_groups_boundaries=eb, 
        # num_of_samples=int(1e6), 
        num_of_samples=int(2e5), 
        # num_of_samples=int(2e4), 
        # num_of_samples=int(2e3), 
        force_detailed_balance=True,
        # force_detailed_balance=False,
        seed=0,
    )
    S_mat = compton_engine.calculate_S_matrix(temperature=T)
    S_mat = np.array(S_mat)
    print(S_mat.shape)

    for e0 in case["ein"]:
        e0 *= units.kev
        g = np.argmin(np.abs(ec-e0))
        plt.stairs(edges=eb/units.kev, values=S_mat[g, :]/(ewid/units.kev)/mbarn ,label=f"$E_{{\\mathrm{{in}}}}$={ec[g]/units.kev:.4g} kev")

    plt.yscale("log")
    plt.ylim(case["ylim"])
    plt.xlim([0., eb[-1]/units.kev])
    plt.grid()
    plt.title(f"T={T/units.kev_kelvin:g}kev")
    plt.xlabel("final photon energy $E$ [kev]")
    plt.ylabel("$\\sigma(E)$ [mbarn/kev]")
    label="Pomranning"
    for file in os.listdir(path.abspath(case["name"])):
        if file.endswith(".txt"):
            sigma_data = np.loadtxt(path.join(case["name"], file), delimiter=",")
            plt.plot(sigma_data[:, 0], sigma_data[:, 1], 'o', markersize=5, c='k', label=label)
            if label != None: label = None
    plt.legend(loc="best", fontsize=15)
    plt.savefig(f"{case["name"]}.png")
    plt.savefig(f"{case["name"]}.pdf")
    plt.show()
    plt.close()