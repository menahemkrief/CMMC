import numpy as np

from matplotlib import pyplot as plt
import sys
import os
from os import path

# add the cpp modules to the system's path
path_to_parent = path.abspath(path.join(path.dirname(path.abspath(__file__)), path.pardir))  
sys.path.append(path_to_parent)

from cpp_modules._compton_matrix_mc import ComptonMatrixMC
import cpp_modules._units as units

def check_pomranning():
    # ------- The Figs. in Pomraning book pages 188-189
    for i, case in enumerate([
                dict(T=1.,  emax=75.,  ein=[5.,10.,20.,40.,60.], ylim=[1,1e4],    data_dir='Pomeraning_1kev_low'),
                dict(T=1.,  emax=340., ein=[80.,120.,200.,300.], ylim=[1e-2,1e2], data_dir='Pomeraning_1kev_high'),
                dict(T=20., emax=140., ein=[5.,10.,20.,40.,60.], ylim=[1e-1,1e3], data_dir='Pomeraning_20kev_low'),
                dict(T=20., emax=140., ein=[10.,20.,40.,60.],    ylim=[1e-1,1e3], data_dir='Pomeraning_20kev_low'),
                dict(T=20., emax=440., ein=[80.,120.,200.,300.], ylim=[1e-2,1e2], data_dir='Pomeraning_20kev_high'),
                ]):
        
        # Units
        barn = 1e-24
        mbarn = 1e-3*barn
        
        # Energy groups boundaries in erg
        eb = np.array(sorted(list(set(
            list(np.linspace(0.01, case["emax"]/10, 50))+\
            list(np.linspace(0.01, case["emax"], 100*4))+\
            list(np.geomspace(0.01, case["emax"], 50))
        ))))
        eb *= units.kev
        
        # energy groups centers
        ec = 0.5*(eb[1:] + eb[:-1])
        # energy groups widths
        ewid = np.diff(eb)
        
        T = case["T"]*units.kev_kelvin

        num_of_samples = int(1e4)

        compton_engine = ComptonMatrixMC(
            energy_groups_centers=ec, 
            energy_groups_boundaries=eb, 
            num_of_samples=num_of_samples,
            force_detailed_balance=True,
        )

        S_mat = np.array(compton_engine.calculate_S_matrix(temperature=T)) # microscopic group-to-group cross section

        for e0 in case["ein"]:
            e0 *= units.kev
            g = np.argmin(np.abs(ec-e0)) # find the energy group closest to e0
            plt.stairs(edges=eb/units.kev, values=S_mat[g, :]/(ewid/units.kev)/mbarn ,label=f"$E_{{\\mathrm{{in}}}}$={ec[g]/units.kev:g} kev")

        for j, file in enumerate(os.listdir(path_to_data_folder:=path.join(path_to_parent, "examples", "data", case['data_dir']))):
            if file.endswith(".txt"):
                sigma_data = np.loadtxt(path.join(path_to_data_folder, file), delimiter=",")
                plt.plot(sigma_data[:, 0], sigma_data[:, 1], 'o', c='k', label="Pomeraning data" if j == 0 else None)

        plt.yscale("log")
        plt.ylim(case["ylim"])
        plt.xlim([0., eb[-1]/units.kev])
        plt.grid()
        plt.title(f"T={T/units.kev_kelvin:g}kev, number of samples={num_of_samples}")
        plt.xlabel("final photon energy $E$ [kev]")
        plt.ylabel("$\\sigma(E)$ [mbarn/kev]")
        plt.legend(loc="best", fontsize=16)
        plt.show()

if __name__ == "__main__":
    check_pomranning()