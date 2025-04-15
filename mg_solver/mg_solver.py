import numpy as np
import scipy.integrate
import sys

import logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger('MGSolver')

# add the cpp modules to the system's path
from os import path
sys.path.append(path.abspath(path.join(path.dirname(path.abspath(__file__)), path.pardir)))
import cpp_modules._units as units
import cpp_modules._planck_integral as planck_integral

def get_planckian(T, eb):
    """
    Returns the frequency integrated Planck distrbution (energy per unit volume) over
    photon energy groups (given by group boundaries `eb` [erg])
    at the given temperature `T` [K].
    """
    return np.asfarray([planck_integral.planck_energy_density_group_integral(E_low=eb[g], E_high=eb[g+1], T=T) for g in range(len(eb)-1)])

class MGSolver():
    """
    An object that solves the multigroup radiation-matter coupled equations, 
    for an infinite homogeneous and isotropic system,
    in the presence of absorption-emission and Compton scattering.

    The system may be moving homologously - its volume can change over time. In that case,
    radiation and material pdv terms, as well as Doppler shift material motion corrections, are
    taken into account.

    All quantities are in c.g.s. units
    """
    def __init__(
            self, *,

            # radiation energy group boundaries [erg]
            energy_groups_boundaries,
            # radiation energy group centers (does not have to be the arithmetic mean of the boundaries) [erg]
            energy_groups_centers,

            # --- initial conditions
            Eg_init,    # initial radiation field (radiation energy per unit volume in each group) [erg/cm^3]
            T_mat_init, # initial material temperature [K]

            # --- material model
            # material EOS - material internal energy per unit volume [erg/cm^3] - as a function of T [K], rho [g/cm^3]
            u_mat_T_eos,

            # material EOS - material temperature [K] - as a function of u [erg/cm^3], rho [g/cm^3]
            T_mat_u_eos,

            # material motion
            rho,                # material mass density [g/cm^3]. If there is no material motion, a constant float should be given. Otherwise, it must be a function of time.
            div_velocity=None,  # the velocity divergence=dln[V(t)]/dt=-dln[rho(t)]/dt, where V(t) is the system's volume. Must be give when material motion is taken into account.
            doppler_term=False, # whether or not to include the Doppler shift term
            gamma_mat=None,     # when there is material motion, the material is heated by PdV. We assume an ideal gas material EOS P(e,rho)=(gamma_mat-1)*rho*e.

            # multigroup absorption opacity - a function of T [K], rho [g/cc]. Returns an array on groups of the absorption macroscopic cross section [1/cm]
            sigma_absorption=None,

            # multigroup Compton matrix - a function of [K], rho [g/cc].  Returns a matrix tau[g][g'] on groups of the Compton macroscopic cross section [1/cm] for photon scattering from g->g'
            tau_compton=None,

            # whether or not to include induced Compton scattering terms
            induced_scattering=True,

            # --- ODE scheme
            ode_scheme="lsoda",
            # ode_scheme="dopri5",
        ):

        logger.info(f"creating an MGSolver calculator...")

        self.energy_groups_boundaries = np.copy(energy_groups_boundaries)
        self.energy_groups_centers = np.copy(energy_groups_centers)
        self.energy_groups_width = np.diff(energy_groups_boundaries)
        self.G = len(self.energy_groups_centers)
        logger.info(f"G={self.G} energy groups")
        assert self.G > 0, self.G
        logger.info(f"Lowest photon energy bin boundary Emin={self.energy_groups_boundaries[0]/units.kev:g} kev")
        logger.info(f"Highest photon energy bin boundary Emax={self.energy_groups_boundaries[-1]/units.kev:g} kev")
        assert len(self.energy_groups_boundaries) == self.G+1
        assert all(self.energy_groups_width>0.)
        assert all(self.energy_groups_boundaries[1:]>self.energy_groups_centers)
        assert all(self.energy_groups_boundaries[:-1]<self.energy_groups_centers)

        self.Eg_init = np.copy(Eg_init)
        assert len(self.Eg_init) == self.G
        
        # --- material motion
        if isinstance(rho, float):
            logger.info(f"material motion OFF. given a constant mass density rho={rho:g} g/cc")
            assert rho > 0.
            self.rho = lambda t: rho
            self.motion = False
            assert div_velocity==None
            assert gamma_mat == None
            assert not doppler_term
        elif callable(rho):
            logger.info(f"material motion ON. given density as a function of time")
            self.rho = rho
            self.motion = True
            assert callable(div_velocity)
            assert gamma_mat != None
            logger.info(f"gamma_mat={gamma_mat:g}")
            logger.info(f"Doppler term {'ON' if doppler_term else 'OFF'}")
        else:
            logger.fatal(f"density must be a function of time or a constant float")
            sys.exit(1)     

        self.div_velocity = div_velocity
        self.doppler_term = doppler_term
        self.gamma_mat = gamma_mat

        self.sigma_absorption = sigma_absorption
        if self.sigma_absorption != None:
            assert callable(self.sigma_absorption)
            assert len(self.sigma_absorption(T_mat_init, rho)) == self.G
        else:
            logger.info("no absorption")

        self.tau_compton = tau_compton
        self.induced_scattering = induced_scattering
        if self.tau_compton != None:
            assert callable(self.tau_compton)
            assert self.tau_compton(T_mat_init, rho).shape == (self.G, self.G)
        else:
            logger.info("no Compton scattering")
        
        self.u_mat_T_eos = u_mat_T_eos
        assert callable(self.u_mat_T_eos)

        self.T_mat_u_eos = T_mat_u_eos
        assert callable(self.T_mat_u_eos)

        self.T_mat_init = T_mat_init
        self.T_rad_init = (np.sum(self.Eg_init)/units.arad)**0.25

        logger.info(f"inital material T={self.T_mat_init/units.kev_kelvin:g} kev")
        logger.info(f"inital radiation effective T={self.T_rad_init / units.kev_kelvin:g} kev")
        logger.info(f"inital material density={self.rho(0.):g} g/cc")

        # ---- auxiliary variables
        self.dydt = np.zeros(self.G+1)
        self.y_init = np.zeros(self.G+1)
        self.y_init[:-1] = np.copy(self.Eg_init)
        self.y_init[-1] = u_mat_T_eos(T_mat_init, rho)

        # ---- set the ODE solver
        logger.info(f"ode_scheme={ode_scheme!r}")
        assert ode_scheme in {"dopri5", "dop853", "lsoda", "zvode", "vode"}
        self.ode_solver = scipy.integrate.ode(self.ydot_func).set_integrator(ode_scheme)
        self.ode_solver.set_initial_value(self.y_init, 0.)

        # ---- The equilibrium temperature is obtained from energy conservation (when there is no motion)
        # it is given by a solution of the nonlinear equation:
        # sum(Eg_init) + u_mat_init = aT_eq^4 + u_mat(T_eq)
        self.T_eq = None
        if not self.motion:
            logger.info("calculating the equilibrium temperature...")
            Etot_init = np.sum(self.Eg_init) + self.u_mat_T_eos(self.T_mat_init, rho)
            # solve the nonlinear equation for the eq. temperature
            self.T_eq = scipy.optimize.fsolve(
                func=lambda T: (units.arad*T**4 + self.u_mat_T_eos(T,rho)-Etot_init),
                x0=0.5*(self.T_rad_init+self.T_mat_init),
            )[0]
            logger.info(f"T_eq={self.T_eq/units.kev_kelvin:.10g} kev")

            # ---- make sure Planckians at maximal/minimal possible material temperatures 
            # ---- are well covered on the minimal-maxiaml group boundaries
            Emin, Emax = energy_groups_boundaries[0], energy_groups_boundaries[-1]
            logger.info(f"Energy groups range E=[{Emin/units.kev:g}, {Emax/units.kev:g}]kev")
            Tmax = max(self.T_eq, self.T_mat_init)
            ekt_1 = Emin/(units.k_boltz*Tmax)
            ekt_2 = Emax/(units.k_boltz*Tmax)
            b_max = planck_integral.planck_integral(a=ekt_1, b=ekt_2)
            logger.info(f"At maximal Tmat={Tmax/units.kev_kelvin:g}kev range E/kTm=[{ekt_1:g}, {ekt_2:g}] covers a Planckian: {b_max*100:g}% (missing {(1-b_max)*100:.4g}%)")
            if 1.-b_max > 1e-4:
                logger.fatal(f"Planckian not covered well at maximal Tmat, enlarge your upper/lower group energies")
                sys.exit(1)
            
            Tmin = min(self.T_eq, self.T_mat_init)
            ekt_1 = Emin/(units.k_boltz*Tmin)
            ekt_2 = Emax/(units.k_boltz*Tmin)
            b_min = planck_integral.planck_integral(a=ekt_1, b=ekt_2)
            logger.info(f"At minimal Tmat={Tmin/units.kev_kelvin:g}kev range E/kTm=[{ekt_1:g}, {ekt_2:g}] covers a Planckian: {b_min*100:g}% (missing {(1-b_min)*100:.4g}%)")
            if 1.-b_max > 1e-4:
                logger.fatal(f"Planckian not covered well at minimal Tmat, enlarge your upper/lower group energies")
                sys.exit(1)
    
    def ydot_func(self, t, y):
        """
        The right hand side of the coupled multigroup radiation-matter equation - 
        the time derivative of the radiation ernergy in each group and the material energy.
        """
        Eg = np.asfarray(y[:-1])
        um = y[-1]
        rho = self.rho(t)
        Tm = self.T_mat_u_eos(um, rho)

        # --- absorption-emission
        abs_term = np.zeros(self.G)
        if self.sigma_absorption != None:
            aTm4bg = get_planckian(Tm, self.energy_groups_boundaries)
            sigma_g = np.asfarray(self.sigma_absorption(Tm, rho))
            abs_term = sigma_g * (aTm4bg-Eg)

        # --- Compton scattering
        compton_term = np.zeros(self.G)
        if self.tau_compton != None:
            tau_g_gp = np.asfarray(self.tau_compton(Tm, rho))
            n = np.zeros(self.G)
            if self.induced_scattering:
                nu = self.energy_groups_centers / units.planck_constant
                dnu = self.energy_groups_width/units.planck_constant
                fac = units.clight**3 / (8.0*np.pi*units.planck_constant)
                n = fac*Eg/(nu**3*dnu)
            
            # in-scatter
            compton_term = np.array([self.energy_groups_centers[g]*(1.+n[g])*np.sum(tau_g_gp[:, g]/self.energy_groups_centers*Eg) for g in range(self.G)])
            # out-scatter
            compton_term -= np.array([Eg[g]*np.sum(tau_g_gp[g, :]*(1.+n)) for g in range(self.G)])

        # ---- Material motion
        pdv_rad = np.zeros(self.G)
        pdv_mat = 0.
        Delta_doppler = np.zeros(self.G)
        if self.motion:
            div = self.div_velocity(t)
            pdv_rad = -4./3.*Eg*div
            pdv_mat = -self.gamma_mat*um*div

            # --- Doppler term
            if self.doppler_term:
                expansion = div >= 0.

                # -----upwind method
                # loop over energy group boundaries
                for gb in range(1, self.G):
                    g_left = gb-1
                    g_right = gb
                    g_donor = g_right if expansion else g_left

                    nuE_boundary = self.energy_groups_centers[g_donor] * Eg[g_donor] / self.energy_groups_width[g_donor]
                    Delta_doppler[g_left] += nuE_boundary
                    Delta_doppler[g_right] -= nuE_boundary
                
                Delta_doppler *= div/3.

        # --- set the time derivatives of the radiation and material energy
        # G radiation group equations
        self.dydt[:-1] = units.clight*(abs_term + compton_term) + pdv_rad + Delta_doppler

        # matter energy equation
        self.dydt[-1] = -units.clight*(np.sum(abs_term + compton_term)) + pdv_mat

        return self.dydt
    
    def solve(self, *, time):
        logger.info(f"t={time/1e-9}ns")
        ysol = self.ode_solver.integrate(time)
        Eg = ysol[:-1]
        u_mat = ysol[-1]
        
        E_rad_tot = np.sum(Eg)
        T_rad = (E_rad_tot/units.arad)**0.25
        T_mat = self.T_mat_u_eos(u_mat, self.rho(time))

        return dict(
            Eg=Eg,
            E_rad_tot=E_rad_tot,
            T_rad=T_rad,
            u_mat=u_mat,
            T_mat=T_mat,
        )