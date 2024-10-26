# CMMC - Compton Matrix Monte Carlo

This code calculates the multigroup Compton scattering kernel on given photon energy bins, and for a given material temperature $T$, mass density $\rho$, average ionization $Z$ and atomic weight $A$.

The integration of the Compton kernel on the energy bins is perofmed using a Monte-Carlo integration.
The Compton matrix $ \tau_{gg'} $ represents the total macroscopic cross-section, which is integrated over the scattering angle, for an incoming photon in group $g$ to scatter to an energy group $g'$.

The coupled radiation-matter equations with Compton scattering (including induced scattering) read:

$$ \frac{d E_{g}}{dt}=ck_{g}\left(b_{g}aT^{4}-E_{g}\right)+\sum_{g'}\tau_{g'g}\frac{\nu_{g}}{\nu_{g'}}\left(1+n_{g}\right)E_{g'}-\sum_{g'}\tau_{gg'}\left(1+n_{g'}\right)E_{g} $$

$$ \frac{d u_{m}}{d t}=-\sum_{g}ck_{g}\left(b_{g}aT^{4}-E_{g}\right)-\sum_{gg'}\tau_{g'g}\frac{\nu_{g}}{\nu_{g'}}\left(1+n_{g}\right)E_{g'}+\sum_{gg'}\tau_{gg'}\left(1+n_{g'}\right)E_{g} $$

where:

* There are $g=1...G$ photon energy groups. $\nu_{g\pm \frac{1}{2}}$ are the frequency bounderies of group $g$, and $\nu_{g}$ is the average photon frequency in group $g$.
* $E_{g}$ is the radiation energy per unit volume in group $g$.
* $u_{m}$ is the material energy per unit volume.
* $T$ is the material temperature.
* $c$ is the speed of light, $a$ is the radiation constant.
* $k_{g}$ is the absorption macroscopic cross section (a.k.a. opacity) in group $g$, with units of inverse length. It is usually defined as the Planck mean of the absorption opacity in the group.
* $b_{g}$ is the fraction of the Planck emission spectra in group $g$:
  
$$ b_{g}=\frac{15}{\pi^{4}}\int_{h\nu_{g-\frac{1}{2}}/k_{B}T}^{h\nu_{g+\frac{1}{2}}/k_{B}T}\frac{x^{3}}{e^{x}-1}dx $$

 where $h$ is the Planck constant. Therefore, when $\nu_{\frac{1}{2}}\ll k_{{B}}T$, $\nu_{G+\frac{1}{2}}\gg k_{{B}}T$, we have $\sum_{g=1}^{G} b_{g}=1$.


* $n_{g}$ is the average photon number in group $g$:
  
$$ n_{g}=\frac{c^{3}E_{g}}{8\pi h\nu_{g}^{3}\Delta\nu_{g}} $$

  where $\Delta\nu_{g}=\nu_{g+ \frac{1}{2}}-\nu_{g- \frac{1}{2}}$.

