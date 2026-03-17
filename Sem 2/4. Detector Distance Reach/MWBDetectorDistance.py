import numpy as np
import matplotlib.pyplot as plt
from MagneticWeberBar import (
    MagneticWeberBar,
    ADMX_EFR,
    DMRADIO_GUT,
    G_NEWTON,
    C_LIGHT,
    HBAR,
    K_B,
    noise_equivalent_strain_broadband,
)

# ─────────────────────────────────────────────────────────────────────────────
# Distance reach for axion superradiance transitions
# ─────────────────────────────────────────────────────────────────────────────

def distance_reach_transition(
        det         : MagneticWeberBar,
        freqs       : np.ndarray,
        h_peak_func : callable,
        tau_func    : callable,
        rho_star    : float = 1.0,
        dist_units  : str   = 'kpc'
) -> np.ndarray:
    """
    Maximum detectable distance for an axion superradiance transition signal.

    The peak strain from a transition process is (input by user):

        h_peak^(trans) = sqrt( 4 G_N / (r^2 omega_t) * (Gamma_sr)^2 / Gamma_t )

    which has the explicit 1/r dependence pulled out as:

        h_peak^(trans) = (1/r) * sqrt( 4 G_N / omega_t * (Gamma_sr)^2 / Gamma_t )

    The signal is a transient of duration tau_t ~ 1/Gamma_sr, so the SNR is:

        SNR = h_peak(r) / sqrt(S_h^noise(f)) * sqrt(tau_t)
            = (1/r) * sqrt(4 G_N / omega_t * (Gamma_sr)^2 / Gamma_t)
                    * sqrt(tau_t / S_h^noise(f))

    Setting SNR = rho_star and solving for r gives d_max directly — no
    reference distance is introduced. This avoids the circularity of
    evaluating h at a reference distance and then rescaling.

    Parameters
    ----------
    det : MagneticWeberBar
        Detector configuration.

    freqs : np.ndarray
        Gravitational wave frequencies to evaluate at [Hz].
        These are the transition frequencies omega_t / 2pi for each
        point in your source parameter sweep.

    h_peak_func : callable
        Function with signature h_peak_func(freqs) -> np.ndarray
        Returns the strain amplitude AT UNIT DISTANCE (r = 1 metre),
        i.e. the factor sqrt(4 G_N / omega_t * (Gamma_sr)^2 / Gamma_t)
        with no 1/r factor included.
        Units: [dimensionless * metres]

    tau_func : callable
        Function with signature tau_func(freqs) -> np.ndarray
        Returns the signal duration tau_t = 1/Gamma_sr at each frequency.
        Units: [seconds]

    rho_star : float
        SNR detection threshold. Default 1.0 for characteristic reach.

    dist_units : str
        Output distance units. One of 'm', 'kpc', 'Mpc', 'pc'.

    Returns
    -------
    d_max : np.ndarray
        Maximum detectable distance at each frequency, in dist_units.
        Shape (len(freqs),)
    """

    # Unit conversion factors to metres
    unit_factors = {
        'm'  : 1.0,
        'pc' : 3.086e16,
        'kpc': 3.086e19,
        'Mpc': 3.086e22,
    }
    if dist_units not in unit_factors:
        raise ValueError(f"dist_units must be one of {list(unit_factors.keys())}")
    unit_conv = unit_factors[dist_units]

    # Noise-equivalent strain PSD at each frequency [Hz^-1]
    S_h_noise = noise_equivalent_strain_broadband(det, freqs)

    # Strain amplitude at unit distance (r=1m) at each frequency
    # h_peak_func should return sqrt(4 G_N / omega_t * Gamma_sr^2 / Gamma_t)
    # Units: dimensionless * metres
    h_unit = h_peak_func(freqs)

    # Signal duration at each frequency [s]
    tau = tau_func(freqs)

    # d_max from SNR = rho_star condition:
    #
    #   rho_star = (1/r) * h_unit * sqrt(tau / S_h_noise)
    #
    #   => r = d_max = h_unit * sqrt(tau / S_h_noise) / rho_star
    #
    # This is in metres — convert to requested units
    d_max_metres = h_unit * np.sqrt(tau / S_h_noise) / rho_star

    return d_max_metres / unit_conv


def plot_distance_reach(
        det           : MagneticWeberBar,
        freqs         : np.ndarray,
        h_peak_func   : callable,
        tau_func      : callable,
        rho_star      : float = 1.0,
        dist_units    : str   = 'kpc',
        label         : str   = '',
        ax            : object = None,
        colour        : str   = 'steelblue',
        show_noise    : bool  = True
):
    """
    Plot the detector distance reach as a function of GW frequency.

    Optionally overlays the noise-equivalent strain on a twin y-axis
    so you can see directly which frequency regime drives the reach.

    Parameters
    ----------
    show_noise : bool
        If True, overlays sqrt(S_h^noise) on a twin y-axis for reference.
    """

    plt.rcParams.update({
        "text.usetex"          : True,
        "font.family"          : "serif",
        "font.serif"           : ["Computer Modern Roman"],
        "text.latex.preamble"  : r"\usepackage{amsmath}"
    })

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    d_max = distance_reach_transition(
        det, freqs, h_peak_func, tau_func, rho_star, dist_units
    )

    ax.loglog(freqs, d_max, color=colour, linewidth=2.0,
              label=label or f'Distance reach ($\\rho^*={rho_star}$)')

    unit_labels = {'m': 'm', 'pc': 'pc', 'kpc': 'kpc', 'Mpc': 'Mpc'}
    ax.set_xlabel(r'$f_t$ [Hz]', fontsize=13)
    ax.set_ylabel(f'$d_{{\\rm max}}$ [{unit_labels[dist_units]}]', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    # Optionally overlay the noise curve on twin axis
    if show_noise:
        ax2 = ax.twinx()
        S_h = noise_equivalent_strain_broadband(det, freqs)
        ax2.loglog(freqs, np.sqrt(S_h),
                   color='gray', linewidth=1.2,
                   linestyle='--', alpha=0.6,
                   label=r'$\sqrt{S_h^{\rm noise}}$')
        ax2.set_ylabel(r'$\left(S_h^{\rm noise}\right)^{1/2}$ [Hz$^{-1/2}$]',
                       fontsize=11, color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        ax2.legend(fontsize=9, loc='lower right')

    plt.tight_layout()
    plt.savefig('distance_reach.png', dpi=150)
    plt.show()
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Example usage — replace these with your actual source functions
# ─────────────────────────────────────────────────────────────────────────────

def h_peak():

    return None

if __name__ == '__main__':

    # ── Your source functions go here ─────────────────────────────────────────
    #
    # h_peak_func(freqs) should return the strain at r = 1 metre:
    #
    #   sqrt( 4 G_N / omega_t * (Gamma_sr)^2 / Gamma_t )
    #
    # tau_func(freqs) should return tau_t = 1 / Gamma_sr
    #
    # Both take a numpy array of frequencies and return a numpy array.
    #
    # The functions below are PLACEHOLDERS showing the expected structure.
    # Replace the bodies with your actual superradiance calculations.

    def h_peak_at_unit_distance(freqs):
        """
        PLACEHOLDER — replace with your actual formula.
        Returns sqrt(4 G_N / omega_t * Gamma_sr^2 / Gamma_t) [m * dimensionless]
        """
        omega_t     = 2 * np.pi * freqs
        Gamma_sr    = 1e-3 * (freqs / 1e3)     # placeholder scaling
        Gamma_t     = 1e-4 * (freqs / 1e3)     # placeholder scaling
        return np.sqrt(4 * G_NEWTON / omega_t * Gamma_sr**2 / Gamma_t)

    def tau_transition(freqs):
        """
        PLACEHOLDER — replace with your actual formula.
        Returns tau_t = 1 / Gamma_sr [s]
        """
        Gamma_sr    = 1e-3 * (freqs / 1e3)     # placeholder scaling
        return 1.0 / Gamma_sr

    # ── Run ───────────────────────────────────────────────────────────────────

    freqs = np.logspace(2, 8, 2000)

    plot_distance_reach(
        det         = ADMX_EFR,
        freqs       = freqs,
        h_peak_func = h_peak_at_unit_distance,
        tau_func    = tau_transition,
        rho_star    = 1.0,
        dist_units  = 'kpc',
        label       = 'ADMX-EFR transition reach',
        show_noise  = True
    )