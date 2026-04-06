"""
GWDetectors.py
==============

Single source of truth for all gravitational-wave detector models used in the
superradiance distance-reach analysis.

Contains
--------
- Physical constants (SI and natural units)
- MagneticWeberBar dataclass  — Weber bar / LC readout detectors
  Instances: ADMX_EFR, DMRADIO_GUT
- IFOConfig dataclass          — Laser interferometer (high-frequency power law)
  Instance dict: IFO_DETECTORS  (key 'adv_ligo')
- All noise PSD functions for both detector families
- plot_all_noise_psds()        — reproduce combined sensitivity curve

Usage
-----
    from GWDetectors import (
        ADMX_EFR, DMRADIO_GUT, IFO_DETECTORS,
        mwb_noise_psd, ifo_noise_psd,
        plot_all_noise_psds,
    )
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Physical constants
# ─────────────────────────────────────────────────────────────────────────────

HBAR     = 1.0546e-34    # J s
E_CHARGE = 1.6022e-19    # C
K_B      = 1.3806e-23    # J K^-1
C_LIGHT  = 3.0e8         # m s^-1
G_NEWTON = 6.674e-11     # m^3 kg^-1 s^-2   (SI)
G_N_NAT  = 6.708e-57     # eV^-2             (natural units, ħ=c=1)
PHI_0    = np.pi * HBAR / E_CHARGE   # Wb, magnetic flux quantum

# Unit conversions (natural <-> SI)
EV_TO_J      = E_CHARGE              # 1 eV in joules
EV_TO_SI     = EV_TO_J / HBAR       # eV -> rad s^-1
INV_EV_TO_M  = HBAR * C_LIGHT / EV_TO_J   # eV^-1 -> m
KPC_TO_M     = 3.086e19             # kpc -> m
MPC_TO_M     = 3.086e22             # Mpc -> m


# ═════════════════════════════════════════════════════════════════════════════
# ① Magnetic Weber Bar detectors
#    (Domcke, Ellis & Rodd 2024/2025, arXiv:2408.01483v2)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class MagneticWeberBar:
    """
    Parameters defining a Magnetic Weber Bar detector.
    All fields in SI units unless stated.

    The effective SQUID-pickup coupling kappa replaces the original
    (alpha^2/4)*(L_squid/L_p) factor throughout (see MagneticWeberBar.py
    for the full derivation).  kappa ~ 1e-2 per Domcke et al.
    """
    # ── Magnet ────────────────────────────────────────────────────────────────
    B0        : float          # Peak DC magnetic field [T]
    ell       : float          # Solenoid length [m]
    r1        : float          # Inner spool radius [m]
    r2        : float          # Outer spool radius [m]
    M         : float          # Magnet mass [kg]
    T_magnet  : float          # Magnet temperature [K]

    # ── Mechanical ────────────────────────────────────────────────────────────
    f_mech    : float          # Mechanical resonant frequency [Hz]
    Q_mech    : float          # Mechanical quality factor

    # ── Pickup loop ───────────────────────────────────────────────────────────
    R_p       : float          # Pickup loop radius [m]

    name      : Optional[str] = None

    # ── SQUID readout ─────────────────────────────────────────────────────────
    kappa     : float = 1.0e-2   # effective SQUID-pickup coupling

    # ── LC resonant readout (optional) ────────────────────────────────────────
    use_LC    : bool  = False
    Q_EM      : float = 2.0e7
    T_LC      : float = 0.01
    f_LC      : float = None     # defaults to f_mech

    # ── Seismic isolation ─────────────────────────────────────────────────────
    n_pendula : int   = 2
    f_pend    : float = 2.0

    # ── Overlap / coupling ────────────────────────────────────────────────────
    eta_210   : float = 0.95
    u_210     : float = 1.0
    G_sq_off  : float = 5.0

    def __post_init__(self):
        self.A_p        = np.pi * self.R_p**2
        self.L_p        = np.pi * self.R_p**2 / self.ell
        self.omega_mech = 2 * np.pi * self.f_mech
        self.ell_eta    = np.sqrt((self.r1**2 + self.r2**2) / 16.0)
        if self.f_LC is None:
            self.f_LC = self.f_mech


# ── Detector instances ────────────────────────────────────────────────────────

ADMX_EFR = MagneticWeberBar(
    B0       = 10.0,
    ell      = 2.0,
    r1       = 0.60,
    r2       = 0.65,
    M        = 40e3,
    T_magnet = 4.0,
    f_mech   = 1.4e3,
    Q_mech   = 1.0e6,
    R_p      = 0.4,
    name     = 'ADMX-EFR',
)

DMRADIO_GUT = MagneticWeberBar(
    B0       = 16.0,
    ell      = 4.0,
    r1       = 0.60,
    r2       = 0.65,
    M        = 40e3,
    T_magnet = 0.02,
    f_mech   = 1.0e3,
    Q_mech   = 1.0e7,
    R_p      = 3.0,
    name     = 'DMRadio-GUT',
)

# Ordered list for sweep loops (det1, det2 convention used downstream)
MWB_DETECTORS = [ADMX_EFR, DMRADIO_GUT]


# ── MWB noise PSD components ──────────────────────────────────────────────────

def _squid_noise_psd(det: MagneticWeberBar) -> float:
    """SQUID flux noise PSD [Wb^2 Hz^-1]. White noise, Eq. S27."""
    return 1.0e-12 * PHI_0**2


def _thermal_force_psd(det: MagneticWeberBar) -> float:
    """Thermal force PSD [N^2 Hz^-1], fluctuation-dissipation theorem."""
    return 2.0 * det.M * K_B * det.T_magnet * det.omega_mech / det.Q_mech


def _thermal_displacement_psd(det: MagneticWeberBar,
                               freqs: np.ndarray) -> np.ndarray:
    """Thermal displacement PSD [m^2 Hz^-1], Lorentzian driven oscillator."""
    omega   = 2 * np.pi * freqs
    S_F     = _thermal_force_psd(det)
    denom   = ((det.omega_mech**2 - omega**2)**2
               + (det.omega_mech * omega / det.Q_mech)**2)
    return (S_F / det.M**2) / denom


def _thermal_flux_noise_psd(det: MagneticWeberBar,
                             freqs: np.ndarray) -> np.ndarray:
    """Thermal mechanical flux noise PSD [Wb^2 Hz^-1], Eq. S29."""
    prefactor = det.kappa**2 * (det.B0 * det.A_p / det.ell)**2
    return prefactor * _thermal_displacement_psd(det, freqs)


def _seismic_displacement_psd(det: MagneticWeberBar,
                               freqs: np.ndarray) -> np.ndarray:
    """Seismic displacement PSD [m^2 Hz^-1] with pendulum suppression."""
    S_seis = 1.0e-18 * np.minimum(1.0, (10.0 / freqs)**4)
    if det.n_pendula > 0:
        suppression = np.where(
            freqs > det.f_pend,
            (det.f_pend / freqs)**(4 * det.n_pendula),
            1.0,
        )
        S_seis *= suppression
    return S_seis


def _gain_factor_sq(det: MagneticWeberBar,
                    freqs: np.ndarray) -> np.ndarray:
    """Dimensionless gain |G(f)|^2. Lorentzian interpolation, Fig. S5."""
    omega   = 2 * np.pi * freqs
    omega_m = det.omega_mech
    lor     = ((omega_m * omega / det.Q_mech)**2 /
               ((omega_m**2 - omega**2)**2
                + (omega_m * omega / det.Q_mech)**2))
    G_sq_on = det.Q_mech**2 * det.G_sq_off
    return det.G_sq_off + (G_sq_on - det.G_sq_off) * lor


def _signal_psd_per_strain(det: MagneticWeberBar,
                            freqs: np.ndarray) -> np.ndarray:
    """Signal transduction S_sig/S_h [Wb^2 strain^{-2} Hz^{-1}], Eq. S25."""
    return det.kappa**2 * det.B0**2 * det.A_p**2 * _gain_factor_sq(det, freqs)


def mwb_noise_psd(det: MagneticWeberBar,
                  freqs: np.ndarray) -> np.ndarray:
    """
    Broadband noise-equivalent strain PSD S_h^noise(f) [Hz^-1].

    S_h^noise = (S_Phi^th + S_Phi^SQ + kappa^2*(B0*Ap/ell)^2*S_x^seis)
                / (S_sig / S_h)
    """
    S_SQ      = _squid_noise_psd(det)
    S_th      = _thermal_flux_noise_psd(det, freqs)
    S_seis    = _seismic_displacement_psd(det, freqs)
    transduct = _signal_psd_per_strain(det, freqs)
    flux_conv = det.kappa**2 * (det.B0 * det.A_p / det.ell)**2
    return (S_SQ + S_th + flux_conv * S_seis) / transduct


def mwb_noise_psd_on_resonance(det: MagneticWeberBar) -> float:
    """
    Noise-equivalent strain PSD exactly at f_mech [Hz^-1].
    Thermal-displacement limited (Eq. S33); independent of EM parameters.
    """
    cl_sq = (det.eta_210 * det.ell_eta * det.u_210)**2
    return (2.0 * K_B * det.T_magnet /
            (det.M * cl_sq * det.omega_mech**3 * det.Q_mech))


def mwb_noise_psd_LC(det: MagneticWeberBar,
                     freqs: np.ndarray) -> np.ndarray:
    """
    Resonant LC noise-equivalent strain PSD [Hz^-1], Eq. S39.
    Only valid above f_mech; returns NaN elsewhere.
    """
    omega_EM = 2 * np.pi * det.f_LC
    S_h_LC   = (4.0 * det.L_p * K_B * det.T_LC /
                (det.Q_EM * omega_EM * det.B0**2 * det.A_p**2))
    result   = np.full_like(freqs, np.nan)
    result[freqs > det.f_mech] = S_h_LC
    return result


# ═════════════════════════════════════════════════════════════════════════════
# ② Laser interferometer — high-frequency power-law model
#    (Schnabel & Korobko 2024, arXiv:2409.03019)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class IFOConfig:
    """
    Laser interferometer described by the Schnabel & Korobko (2024)
    high-frequency power-law sensitivity extrapolation.

    sqrt(S_h(f)) = asd_anchor * (f / f_FSR)   for f >= f_FSR
    """
    name       : str
    arm_length : float    # effective arm / resonator length [m]
    asd_anchor : float    # sqrt(S_h) at first FSR tooth [Hz^{-1/2}]
    n_max      : int      # highest comb index to consider
    color      : str
    linestyle  : str = '-'

    def __post_init__(self):
        self.f_FSR = C_LIGHT / (2.0 * self.arm_length)   # [Hz]


# ── IFO detector catalogue ────────────────────────────────────────────────────

IFO_DETECTORS = {
    'adv_ligo': IFOConfig(
        name       = r'LIGO HF',
        arm_length = 4.0e3,
        asd_anchor = 4.0e-23,
        n_max      = 1000,
        color      = 'purple',
        linestyle  = '-',
    ),
}


# ── IFO noise PSD functions ───────────────────────────────────────────────────

def ifo_noise_psd(freqs: np.ndarray,
                  detector_key: str = 'adv_ligo') -> np.ndarray:
    """
    Noise-equivalent strain PSD S_h(f) [Hz^-1] for the named IFO.

    Returns NaN below f_FSR (model not valid there).
    """
    ifo    = IFO_DETECTORS[detector_key]
    freqs  = np.asarray(freqs, dtype=float)
    S_h    = np.full_like(freqs, np.nan)
    mask   = freqs >= ifo.f_FSR
    S_h[mask] = (ifo.asd_anchor * freqs[mask] / ifo.f_FSR)**2
    return S_h


def ifo_asd_envelope(ifo: IFOConfig,
                     freqs: np.ndarray) -> np.ndarray:
    """ASD envelope sqrt(S_h) = A_det * f/f_FSR [Hz^{-1/2}]."""
    return ifo.asd_anchor * (freqs / ifo.f_FSR)


# ═════════════════════════════════════════════════════════════════════════════
# ③ Combined sensitivity plot
# ═════════════════════════════════════════════════════════════════════════════

def plot_all_noise_psds(
        f_min: float = 1e1,
        f_max: float = 1e9,
        n_pts: int   = 4000,
        plot_LC      : bool = False,
        plot_BBN     : bool = False,
        savepath     : str  = None,
):
    """
    Plot noise-equivalent strain ASD for all detectors on one figure.

    Covers both the MWB detectors (ADMX-EFR, DMRadio-GUT) and the
    LIGO high-frequency power-law envelope (adv_ligo).

    Parameters
    ----------
    f_min, f_max : float
        Frequency range [Hz].
    n_pts : int
        Number of points in the frequency grid.
    plot_LC : bool
        If True, also draw the LC resonant sensitivity for MWB detectors
        where use_LC = True.
    plot_BBN : bool
        If True, overlay the cosmological BBN bound on stochastic GW background.
    savepath : str, optional
        If given, save the figure to this path (PDF recommended).
    """

    plt.rcParams.update({
        "text.usetex"        : True,
        "font.family"        : "serif",
        "font.serif"         : ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    freqs = np.logspace(np.log10(f_min), np.log10(f_max), n_pts)
    fig, ax = plt.subplots(figsize=(5, 4))

    # ── MWB detectors ─────────────────────────────────────────────────────────
    mwb_colors = {'ADMX-EFR': 'steelblue', 'DMRadio-GUT': 'teal'}

    for det in MWB_DETECTORS:
        col  = mwb_colors.get(det.name, 'gray')
        S_h  = mwb_noise_psd(det, freqs)
        ax.loglog(freqs, np.sqrt(S_h),
                  color=col, linewidth=2.0, label=det.name)

        if plot_LC and det.use_LC:
            S_h_LC = mwb_noise_psd_LC(det, freqs)
            mask   = ~np.isnan(S_h_LC)
            ax.loglog(freqs[mask], np.sqrt(S_h_LC[mask]),
                      color=col, linewidth=1.5, linestyle='--',
                      label=fr'{det.name} (LC)')

    # ── IFO detectors ─────────────────────────────────────────────────────────
    for key, ifo in IFO_DETECTORS.items():
        f_valid = freqs[freqs >= ifo.f_FSR]
        if len(f_valid) == 0:
            continue
        asd = ifo_asd_envelope(ifo, f_valid)
        ax.loglog(f_valid, asd,
                  color=ifo.color, linewidth=1.8,
                  linestyle=ifo.linestyle, label=ifo.name)
        # Mark the FSR frequency
        ax.axvline(ifo.f_FSR, color=ifo.color,
                   linewidth=0.7, linestyle=':', alpha=0.4)

    # ── BBN bound ─────────────────────────────────────────────────────────────
    if plot_BBN:
        H0    = 67.4e3 / 3.086e22   # s^-1
        Omega = 1.8e-6
        S_BBN = (3.0 * H0**2 / (2.0 * np.pi**2)) * Omega / freqs**3
        ax.loglog(freqs, np.sqrt(S_BBN),
                  color='darkgreen', linewidth=1.2,
                  linestyle=':', label='BBN bound')

    ax.set_xlabel(r'$f\ [\mathrm{Hz}]$', fontsize=13)
    ax.set_ylabel(
        r'$\left(S_h^\mathrm{noise}\right)^{1/2}\ [\mathrm{Hz}^{-1/2}]$',
        fontsize=13,
    )
    ax.set_xlim(f_min, f_max)
    ax.set_ylim(1e-26, 1e-14)
    ax.legend(fontsize=9, loc='upper left', frameon=False)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
        print(f'[saved] {savepath}')

    plt.show()
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Run standalone: plot combined sensitivity
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # Optionally enable LC readout for DMRadio-GUT
    DMRADIO_GUT.use_LC = True
    DMRADIO_GUT.T_LC   = 0.01
    DMRADIO_GUT.Q_EM   = 2.0e7

    plot_all_noise_psds(
        f_min    = 1e1,
        f_max    = 1e9,
        plot_LC  = False,
        plot_BBN = False,
        savepath = '4. Detector Distance Reach/detector_noise_psds.pdf',
    )
