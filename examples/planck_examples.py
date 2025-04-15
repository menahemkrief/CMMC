import numpy as np
import sys
import scipy.integrate

# add the cpp modules to the system's path
from os import path
sys.path.append(path.abspath(path.join(path.dirname(path.abspath(__file__)), path.pardir)))
import cpp_modules._units as units
import cpp_modules._planck_integral as planck_integral

from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('classic')
matplotlib.rcParams.update({
    'font.size': 17,        # Default font size
})

normalized_planck = lambda x: 15.*x*x*x*np.exp(-x)/(np.pi*np.pi*np.pi*np.pi*(1-np.exp(-x)))

planck_integral_exact = np.vectorize(
    lambda a, b: scipy.integrate.quad(normalized_planck, a, b)[0]
)

def check_dimensionless_integral():

    for xb in [

        np.array([1e-10,1e-8,1e-5,1e-3,1e-2,0.05,0.5,0.7,1.1,2.,2.5,3.1,3.8,4.4,5.1,6.,7.,15.,20.,50.,100.]),

        np.array(sorted(set(
        list(np.linspace(1e-10, 1e2, 100)) + \
        list(np.geomspace(1e-10,1e2, 100))
        ))),
        ]:

        xc = 0.5*(xb[1:]+xb[:-1])
        G = len(xc)
        int_aprx = np.array([planck_integral.planck_integral(a=xb[i], b=xb[i+1]) for i in range(G)])
        int_ex = np.array([planck_integral_exact(xb[i], xb[i+1]) for i in range(G)])

        print("sum", sum(int_aprx))
        for i in range(G):
            print(f"G={i+1} [x1,x2]=[{xb[i]:.5g} {xb[i+1]:.5g}] approx={int_aprx[i]} exact={int_ex[i]} err={abs(int_aprx[i]/int_ex[i]-1.):g}")

        dx = np.diff(xb)
        plt.figure("pl")
        plt.plot(xc, normalized_planck(xc), "-ko", label="Planckian $b(x)$")
        plt.stairs(edges=xb, values=int_ex/dx, color="b", lw=2., linestyle="-", label="Exact avg. (integral)")
        plt.stairs(edges=xb, values=int_aprx/dx, color="r", lw=2., linestyle="--", label="Clark avg. (integral)")
        plt.legend(loc="best")
        plt.grid()
        plt.ylabel("Normalized Planck function $b(x)$")
        plt.xlabel("$x$")
        plt.yscale("log")
        plt.xscale("log")

        plt.figure("err")
        plt.plot(xc, np.abs(int_aprx/int_ex-1.), ls="--")
        plt.ylabel("Clark vs. quad integrals error at each bin")
        plt.grid()
        plt.xlabel("$x$")
        plt.xscale("log")
        plt.yscale("log")

        plt.show()

def check_dimensional_integral():
    # some prints
    print(planck_integral.planck_integral(a=20,b=300.))
    print(planck_integral.planck_integral(a=1e-4,b=3000.))

    T = 50*units.kev_kelvin
    E_low=0.001*units.kev
    E_high=500*units.kev

    Eg = planck_integral.planck_energy_density_group_integral(E_low=E_low, E_high=E_high, T=T)
    aT4 = units.arad*T**4
    print(Eg)
    print(aT4)
    print(Eg/aT4)
    print(planck_integral.planck_integral(a=E_low/(units.k_boltz*T),b=E_high/(units.k_boltz*T)))

    # ---- compare the group integrated Planck spectrum to the pointwise Planck spectra
    T = 10*units.kev_kelvin
    E_low=0.01*units.kev
    E_high=300.*units.kev
    G = 32
    eb = np.geomspace(E_low, E_high, G+1)
    ec = 0.5*(eb[1:]+eb[:-1])
    Eg = np.asfarray([planck_integral.planck_energy_density_group_integral(E_low=eb[g], E_high=eb[g+1], T=T) for g in range(len(eb)-1)])

    plt.stairs(edges=eb/units.kev, values=Eg/(np.diff(eb)/units.kev), color="r", label="$E_{{g}}/\\Delta E_{{g}}$")

    plt.plot(ec/units.kev, normalized_planck(ec/(units.k_boltz*T))*units.arad*T**4/(units.k_boltz*T/units.kev), "--bo", label=f"Planckian")

    plt.legend(loc="best", fontsize=16)
    plt.yscale("log")
    plt.xscale("log")
    plt.xlim([eb[0]/units.kev, eb[-1]/units.kev])
    plt.grid()
    plt.title(f"T={T/units.kev_kelvin:g}kev")
    plt.xlabel("photon energy $E$ [kev]")
    plt.ylabel("$U(E) \\ [\\mathrm{{erg/cm^{{3}}/kev}}]$ ")
    plt.show()

if __name__ == "__main__":
    check_dimensionless_integral()
    check_dimensional_integral()