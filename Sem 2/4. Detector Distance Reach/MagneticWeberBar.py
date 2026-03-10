import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Callable

# ─────────────────────────────────────────────────────────────────────────────
# Physical constants
# ─────────────────────────────────────────────────────────────────────────────

HBAR        = 1.0546e-34   # J s
E_CHARGE    = 1.6022e-19   # C
K_B         = 1.3806e-23   # J / K
PHI_0       = np.pi * HBAR / E_CHARGE   # Wb, magnetic flux quantum
G_NEWTON    = 6.674e-11    # m^3 kg^-1 s^-2
C_LIGHT     = 3.0e8        # m / s

# ─────────────────────────────────────────────────────────────────────────────
# Detector configuration dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MagneticWeberBar:
    """
    Parameters defining a Magnetic Weber Bar detector configuration.
    All SI units unless stated.

    Parameters defined from Fig. 1 caption and SM Section S.IV of
    Domcke, Ellis & Rodd (2024/2025), arXiv:2408.01483v2.
    """
    # ── Magnet ────────────────────────────────────────────────────────────────
    B0          : float   # Peak DC magnetic field [T]
    ell         : float   # Solenoid length [m]
    r1          : float   # Inner spool radius [m]
    r2          : float   # Outer spool radius [m]
    M           : float   # Magnet mass [kg]
    T_magnet    : float   # Magnet temperature [K]

    # ── Mechanical ────────────────────────────────────────────────────────────
    f_mech      : float   # Mechanical resonant frequency [Hz]
    Q_mech      : float   # Mechanical quality factor

    # ── Pickup loop ───────────────────────────────────────────────────────────
    R_p         : float   # Pickup loop radius [m]

    # ── SQUID readout ─────────────────────────────────────────────────────────
    alpha       : float = 1.0 / np.sqrt(2)   # SQUID coupling coefficient
    L_squid     : float = 1.0e-9             # SQUID inductance [H]

    # ── LC resonant readout (optional) ────────────────────────────────────────
    use_LC      : bool  = False
    Q_EM        : float = 2.0e7    # EM quality factor of LC circuit
    T_LC        : float = 0.01     # LC circuit temperature [K]
    f_LC        : float = None     # LC resonant frequency [Hz]; defaults to f_mech

    # ── Seismic isolation ─────────────────────────────────────────────────────
    n_pendula   : int   = 2        # Number of pendulum stages (0 = no isolation)
    f_pend      : float = 2.0      # Pendulum natural frequency [Hz]

    # ── Overlap / coupling (from SM Section S.I and S.II) ────────────────────
    eta_210     : float = 0.95     # GW-mode overlap factor (Eq. S16, Fig. S2)
    u_210       : float = 1.0      # Eigenmode amplitude (dimensionless, ~1)
    G_sq_off    : float = 5.0      # |G|^2 off resonance (Fig. S5)

    def __post_init__(self):
        # Derived quantities
        self.A_p    = np.pi * self.R_p**2          # Pickup loop area [m^2]
        self.L_p    = np.pi * self.R_p**2 / self.ell  # Pickup loop inductance [H]
        self.omega_mech = 2 * np.pi * self.f_mech  # [rad/s]

        # ell_eta: characteristic length scale for GW-mode coupling (Eq. S16)
        # For GW along z-axis (theta=0):
        #   ell_eta^2 = [ell^2 + 9(r1^2+r2^2) + (3(r1^2+r2^2)-ell^2)*cos(0)] / 192
        #             = [ell^2 + 9(r1^2+r2^2) + 3(r1^2+r2^2) - ell^2] / 192
        #             = 12(r1^2+r2^2) / 192
        #             = (r1^2+r2^2) / 16
        self.ell_eta = np.sqrt((self.r1**2 + self.r2**2) / 16.0)

        if self.f_LC is None:
            self.f_LC = self.f_mech


# ─────────────────────────────────────────────────────────────────────────────
# ADMX-EFR and DMRadio-GUT configurations from Fig. 1 caption
# ─────────────────────────────────────────────────────────────────────────────

ADMX_EFR = MagneticWeberBar(
    B0        = 10.0,
    ell       = 2.0,
    r1        = 0.60,
    r2        = 0.65,
    M         = 40e3,          # 40 tonnes
    T_magnet  = 4.0,
    f_mech    = 1.4e3,
    Q_mech    = 1.0e6,
    R_p       = 0.4,
)

DMRADIO_GUT = MagneticWeberBar(
    B0        = 16.0,
    ell       = 4.0,
    r1        = 0.60,
    r2        = 0.65,
    M         = 40e3,
    T_magnet  = 0.02,
    f_mech    = 1.0e3,
    Q_mech    = 1.0e7,
    R_p       = 3.0,
)


# ─────────────────────────────────────────────────────────────────────────────
# Noise PSD components
# All return flux noise in Wb^2/Hz
# ─────────────────────────────────────────────────────────────────────────────

def squid_noise_psd(det: MagneticWeberBar) -> float:
    """
    SQUID flux noise PSD — Eq. S27.
    Flat (white noise) above ~10 Hz.
    From experimental SQUID literature (Refs [34-36] of Domcke et al.)

    Returns: S_Phi^SQ [Wb^2/Hz]
    """
    return 1.0e-12 * PHI_0**2   # ~ 4.28e-42 Wb^2/Hz


def thermal_force_psd(det: MagneticWeberBar) -> float:
    """
    Thermal force PSD on the magnet from fluctuation-dissipation theorem.
    S_F^th = 2 M T omega_mech / Q_mech
    Kubo (1966), Saulson (1990) — Refs [37,38] of Domcke et al.

    Returns: S_F^th [N^2/Hz]
    """
    return 2.0 * det.M * K_B * det.T_magnet * det.omega_mech / det.Q_mech


def thermal_displacement_psd(det: MagneticWeberBar,
                              freqs: np.ndarray) -> np.ndarray:
    """
    Thermal displacement PSD of the magnet — Eq. S28.
    Lorentzian driven oscillator response.

    S_x^th(omega) = S_F^th * M^{-2} / [(omega_mech^2 - omega^2)^2
                                         + (omega_mech * omega / Q_mech)^2]

    Returns: S_x^th [m^2/Hz], shape (len(freqs),)
    """
    omega       = 2 * np.pi * freqs
    S_F         = thermal_force_psd(det)
    numerator   = S_F / det.M**2
    denominator = ((det.omega_mech**2 - omega**2)**2
                   + (det.omega_mech * omega / det.Q_mech)**2)
    return numerator / denominator


def thermal_flux_noise_psd(det: MagneticWeberBar,
                            freqs: np.ndarray) -> np.ndarray:
    """
    Thermal mechanical flux noise PSD — Eq. S29.
    Converts thermal displacement to flux noise seen by SQUID.

    S_Phi^th(omega) ~ (alpha^2/4) * (L/Lp) * (B0*Ap/ell)^2 * S_x^th(omega)

    Note: this is an approximation away from resonance. On resonance we
    use the direct displacement comparison (Eq. S33) instead — see
    noise_equivalent_strain_on_resonance().

    Returns: S_Phi^th [Wb^2/Hz], shape (len(freqs),)
    """
    prefactor   = (det.alpha**2 / 4.0) * (det.L_squid / det.L_p)
    prefactor  *= (det.B0 * det.A_p / det.ell)**2
    return prefactor * thermal_displacement_psd(det, freqs)


def seismic_displacement_psd(det: MagneticWeberBar,
                              freqs: np.ndarray) -> np.ndarray:
    """
    Seismic displacement PSD with pendulum suppression — Eq. S31.

    Quiet site: S_x^seismic ~ 1e-18 m^2/Hz * min[1, (10Hz/f)^4]
    Each pendulum stage suppresses by (f_pend/f)^4 above f_pend.

    Returns: S_x^seismic [m^2/Hz], shape (len(freqs),)
    """
    # Quiet site seismic floor
    S_seismic = 1.0e-18 * np.minimum(1.0, (10.0 / freqs)**4)

    # Pendulum suppression
    if det.n_pendula > 0:
        suppression = np.where(
            freqs > det.f_pend,
            (det.f_pend / freqs)**(4 * det.n_pendula),
            1.0
        )
        S_seismic *= suppression

    return S_seismic


# ─────────────────────────────────────────────────────────────────────────────
# Signal PSD and gain factor
# ─────────────────────────────────────────────────────────────────────────────

def gain_factor_sq(det: MagneticWeberBar, freqs: np.ndarray) -> np.ndarray:
    """
    Dimensionless gain factor |G(f)|^2 — defined below Eq. (6) / Eq. S24.

    This is a placeholder for the full numerical Biot-Savart computation
    (Eqs. S20/S21 integrated over the pickup loop area).

    Current approximation (from Fig. S5 of Domcke et al.):
      - Off resonance:   |G|^2 = G_sq_off ~ 5
      - On resonance:    |G|^2 ~ Q_mech^2
      - Transition modelled as a Lorentzian interpolation

    NOTE: Replace this function with the full numerical Biot-Savart
    integral once the eigenmode calculation is implemented.

    Returns: |G|^2 [dimensionless], shape (len(freqs),)
    """
    omega       = 2 * np.pi * freqs
    omega_m     = det.omega_mech

    # Lorentzian lineshape centred on mechanical resonance
    lorentzian  = (omega_m * omega / det.Q_mech)**2 / (
                   (omega_m**2 - omega**2)**2
                   + (omega_m * omega / det.Q_mech)**2)

    # Interpolate between off-resonance value and Q^2 peak
    G_sq_on     = det.Q_mech**2 * det.G_sq_off
    return det.G_sq_off + (G_sq_on - det.G_sq_off) * lorentzian


def signal_psd_per_strain(det: MagneticWeberBar,
                           freqs: np.ndarray) -> np.ndarray:
    """
    Signal PSD divided by GW strain PSD: S_sig(f) / S_h(f) — Eq. S25.

    S_sig / S_h = (alpha^2 / 4) * (L / L_p) * B0^2 * A_p^2 * |G|^2

    This is the transduction efficiency of the detector.

    Returns: S_sig/S_h [Wb^2/Hz per (strain^2/Hz)], shape (len(freqs),)
    """
    prefactor   = (det.alpha**2 / 4.0) * (det.L_squid / det.L_p)
    prefactor  *= det.B0**2 * det.A_p**2
    return prefactor * gain_factor_sq(det, freqs)


# ─────────────────────────────────────────────────────────────────────────────
# Noise-equivalent strain PSD
# ─────────────────────────────────────────────────────────────────────────────

def noise_equivalent_strain_on_resonance(det: MagneticWeberBar) -> float:
    """
    Noise-equivalent strain PSD exactly on the mechanical resonance — Eq. S33.

    (S_h^noise)_mech = 2 T / [M * (eta_210 * ell_eta * u_210)^2
                                 * omega_mech^3 * Q_mech]

    This bypasses the flux/gain calculation entirely by directly comparing
    the GW-induced displacement PSD to the thermal displacement PSD.
    Identical in form to a classical Weber bar — no EM parameters appear.

    Returns: S_h^noise at f_mech [Hz^{-1}]
    """
    coupling_length_sq  = (det.eta_210 * det.ell_eta * det.u_210)**2
    return (2.0 * K_B * det.T_magnet /
            (det.M * coupling_length_sq
             * det.omega_mech**3 * det.Q_mech))


def noise_equivalent_strain_broadband(det: MagneticWeberBar,
                                       freqs: np.ndarray) -> np.ndarray:
    """
    Broadband noise-equivalent strain PSD — Eq. S32.

    S_h^noise(f) = [S_Phi^th(f) + S_Phi^SQ] / [S_sig(f) / S_h(f)]

    On resonance we override with the exact Weber bar expression (Eq. S33)
    to avoid the approximation in thermal_flux_noise_psd().

    Returns: S_h^noise(f) [Hz^{-1}], shape (len(freqs),)
    """
    S_SQ        = squid_noise_psd(det)
    S_th        = thermal_flux_noise_psd(det, freqs)
    S_seis      = seismic_displacement_psd(det, freqs)
    transduct   = signal_psd_per_strain(det, freqs)

    # Combined noise (flux units)
    # Seismic is added as an effective displacement noise converted to flux
    flux_conv   = ((det.alpha**2 / 4.0) * (det.L_squid / det.L_p)
                   * (det.B0 * det.A_p / det.ell)**2)
    S_noise_tot = S_SQ + S_th + flux_conv * S_seis

    # Noise-equivalent strain
    S_h_noise   = S_noise_tot / transduct

    # Override on resonance with exact Eq. S33
    i_res       = np.argmin(np.abs(freqs - det.f_mech))
    S_h_noise[i_res] = noise_equivalent_strain_on_resonance(det)

    return S_h_noise


def noise_equivalent_strain_LC(det: MagneticWeberBar,
                                freqs: np.ndarray) -> np.ndarray:
    """
    Noise-equivalent strain for resonant LC readout — Eq. S39.

    Above the mechanical resonance:
      (S_h^noise)^LC ~ 4 L_p T_LC / (Q_EM * omega_EM * B0^2 * A_p^2)

    This is frequency-independent (flat) above f_mech, improving over
    the broadband SQUID floor by factor ~ Q_EM * omega_EM * S_Phi^SQ
                                                / (alpha^2 * T_LC * L)

    Returns: S_h^noise_LC(f) [Hz^{-1}], shape (len(freqs),)
    """
    omega_EM    = 2 * np.pi * det.f_LC
    S_h_LC      = (4.0 * det.L_p * K_B * det.T_LC
                   / (det.Q_EM * omega_EM * det.B0**2 * det.A_p**2))

    # Only apply above mechanical resonance
    result      = np.full_like(freqs, np.nan)
    mask        = freqs > det.f_mech
    result[mask] = S_h_LC

    return result


# ─────────────────────────────────────────────────────────────────────────────
# GW strain input interface
# ─────────────────────────────────────────────────────────────────────────────

def snr_broadband(det: MagneticWeberBar,
                  freqs: np.ndarray,
                  h_strain_psd: np.ndarray,
                  T_obs: float) -> float:
    """
    Matched-filter SNR for a broadband source — Eq. (6).

    SNR^2 = 2 * T_obs * integral[ (S_h_signal / S_h_noise)^2 df ]

    Parameters
    ----------
    h_strain_psd : np.ndarray
        GW strain PSD S_h(f) of your source [Hz^{-1}].
        THIS IS YOUR INPUT — provide from your own source calculations.
    T_obs : float
        Observation time [s].

    Returns
    -------
    SNR : float
    """
    S_h_noise   = noise_equivalent_strain_broadband(det, freqs)
    integrand   = (h_strain_psd / S_h_noise)**2
    return np.sqrt(2.0 * T_obs * np.trapz(integrand, freqs))


def snr_monochromatic(det: MagneticWeberBar,
                      freq: float,
                      h0: float,
                      T_obs: float) -> float:
    """
    SNR for a persistent monochromatic source at a single frequency.

    SNR = h0 / sqrt(S_h^noise(f)) * sqrt(T_obs)

    Parameters
    ----------
    freq : float
        Source frequency [Hz].
    h0 : float
        GW strain amplitude [dimensionless].
        THIS IS YOUR INPUT — provide from your own source calculations.
    T_obs : float
        Observation time [s].

    Returns
    -------
    SNR : float
    """
    freqs       = np.array([freq])
    S_h_noise   = noise_equivalent_strain_broadband(det, freqs)[0]
    return h0 / np.sqrt(S_h_noise) * np.sqrt(T_obs)


def max_distance(det: MagneticWeberBar,
                 freqs: np.ndarray,
                 h_at_1Mpc: np.ndarray,
                 T_obs: float,
                 snr_threshold: float = 1.0) -> np.ndarray:
    """
    Maximum detectable distance as a function of frequency — Eq. (7/8).

    d_max(f) = h(f, d=1Mpc) * sqrt(T_obs) / (SNR* * sqrt(S_h^noise(f)))

    Parameters
    ----------
    h_at_1Mpc : np.ndarray
        GW strain amplitude at reference distance of 1 Mpc [dimensionless].
        THIS IS YOUR INPUT — provide from your own source calculations.
        Must have same shape as freqs.
    T_obs : float
        Observation time [s].
    snr_threshold : float
        Detection SNR threshold (default 1.0 for characteristic reach).

    Returns
    -------
    d_max : np.ndarray
        Maximum detectable distance [Mpc], shape (len(freqs),)
    """
    S_h_noise   = noise_equivalent_strain_broadband(det, freqs)
    return (h_at_1Mpc * np.sqrt(T_obs)
            / (snr_threshold * np.sqrt(S_h_noise)))


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_sensitivity(detectors: dict,
                     f_min: float = 1e2,
                     f_max: float = 1e8,
                     n_points: int = 2000,
                     plot_LC: bool = True,
                     plot_BBN: bool = True):
    """
    Reproduce Fig. 1 of Domcke et al. — noise-equivalent strain ASD.

    Parameters
    ----------
    detectors : dict
        Keys are labels, values are MagneticWeberBar instances.
    """
    freqs   = np.logspace(np.log10(f_min), np.log10(f_max), n_points)

    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}"
    })

    fig, ax = plt.subplots(figsize=(5, 4))

    colours = {'ADMX-EFR': 'steelblue', 'DMRadio-GUT': 'teal'}

    for label, det in detectors.items():
        col = colours.get(label, 'gray')

        # Broadband sensitivity
        S_h     = noise_equivalent_strain_broadband(det, freqs)
        ax.loglog(freqs, np.sqrt(S_h),
                  color=col, linewidth=2.0,
                  label=f'{label} Broadband')

        # LC resonant sensitivity
        if plot_LC and det.use_LC:
            S_h_LC  = noise_equivalent_strain_LC(det, freqs)
            mask    = ~np.isnan(S_h_LC)
            ax.loglog(freqs[mask], np.sqrt(S_h_LC[mask]),
                      color=col, linewidth=1.5, linestyle='--',
                      label=f'{label} LC Resonant')

    # BBN bound — Omega_GW < 1.8e-6, H0 = 67.4 km/s/Mpc
    if plot_BBN:
        H0      = 67.4e3 / 3.086e22        # s^{-1}
        Omega   = 1.8e-6
        S_BBN   = (3.0 * H0**2 / (2.0 * np.pi**2)) * Omega / freqs**3
        ax.loglog(freqs, np.sqrt(S_BBN),
                  color='darkgreen', linewidth=1.2,
                  linestyle=':', label='BBN bound')

    ax.set_xlabel(r'$f$ [Hz]', fontsize=13)
    ax.set_ylabel(r'$\left(S_h^\mathrm{noise}\right)^{1/2}$ [Hz$^{-1/2}$]',
                  fontsize=13)
    # ax.set_title('Magnetic Weber Bar — Noise-Equivalent Strain Sensitivity',                 fontsize=13)
    ax.set_xlim(f_min, f_max)
    ax.set_ylim(1e-26, 1e-14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('mwb_sensitivity.png', dpi=150)
    plt.show()
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # Enable LC readout for DMRadio
    DMRADIO_GUT.use_LC  = True
    DMRADIO_GUT.T_LC    = 0.01
    DMRADIO_GUT.Q_EM    = 2.0e7

    detectors = {
        'ADMX-EFR'    : ADMX_EFR,
        'DMRadio-GUT' : DMRADIO_GUT,
    }

    plot_sensitivity(detectors)

    # ── Example: plug in your own GW strain ───────────────────────────────────
    # Replace these with your own source calculations.
    # h_at_1Mpc should be the strain amplitude your source produces at 1 Mpc.

    # freqs_example   = np.logspace(2, 8, 2000)
    # T_obs           = 10.0 * 86400.0   # 10 days in seconds

    # PLACEHOLDER — replace with your source model
    # h_at_1Mpc_placeholder = np.ones_like(freqs_example) * 1e-20

    # d_max = max_distance(ADMX_EFR, freqs_example,
    #                      h_at_1Mpc_placeholder, T_obs)

    # print(f"Peak distance reach: {np.nanmax(d_max):.3e} Mpc")
    # print(f"At frequency:        {freqs_example[np.nanargmax(d_max)]:.3e} Hz")