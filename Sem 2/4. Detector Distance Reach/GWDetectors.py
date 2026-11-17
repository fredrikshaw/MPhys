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
- TABULATED_DETECTORS        — published reference curves in eight detector families
- SHARK_FIN_DETECTORS         — the ten supplied binary-distance curves
- PHOTON_REGENERATION_DETECTORS — ALPS II, JURA, OSQAR II, CAST and IAXO
- detector_noise_psd()        — unified noise access by object or catalogue key
- plot_all_noise_psds()       — combined PSD/ASD; photon-conversion curves excluded by default
- plot_shark_fin_horizons()   — imported distance curves in their original units

Usage
-----
    from GWDetectors import (
        ADMX_EFR, DMRADIO_GUT, IFO_DETECTORS,
        mwb_noise_psd, ifo_noise_psd,
        TABULATED_DETECTORS, detector_noise_psd, plot_all_noise_psds,
    )

    # New detectors: all frequencies in Hz, output PSD in Hz^-1.
    psd = detector_noise_psd('dmradio_m3_broad', np.array([1e7, 1e8]))
    fig, ax = plot_all_noise_psds(show=False)  # PSD; quantity='asd' also works

The tabulated catalogue supplies noise and binary-horizon data. Response profiles
in data/detector_response_models.json distinguish published component parameters
from explicit hypothetical receiver assumptions. Ring-up times are component
amplitude time constants, not minimum detectable source lifetimes.
"""

import numpy as np
import matplotlib
if __name__ == '__main__':
    # Standalone runs export files and do not need a desktop plotting backend.
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, replace
from functools import cached_property, lru_cache
from typing import ClassVar, Optional
from scipy.interpolate import interp1d
from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter
from matplotlib import patheffects
from DetectorResponse import load_response_models, ResponseComponent

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


# ────────────────────────────────────────────────────────────────────────────
# Load in the LIGO data
# ────────────────────────────────────────────────────────────────────────────

def load_ligo_psd(filepath):
    """
    Load LIGO noise PSD file.

    Expected format:
        f [Hz]   S_h [Hz^-1]   (Hanford is column 2 in LIGO-NESPSD.dat).
        This loader does not accept ASD without an explicit conversion.

    Returns
    -------
    interp function: f -> S_h(f)
    """
    data = np.loadtxt(filepath)

    f   = data[:, 0]
    S_h = data[:, 1]

    return interp1d(
        f, S_h,
        bounds_error=False,
        fill_value=np.nan
    )

current_dir = Path(__file__).resolve().parent
LIGO_PSD_interp = load_ligo_psd(current_dir / "LIGO-NESPSD.dat")

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

    def ring_up_time(self, component=None) -> float:
        """Component amplitude time [s]; default is 2 Q_mech / omega_mech.

        This is the mechanical resonance envelope time, not a global broadband
        cutoff. Use component='receiver_envelope' for the assumed readout pole.
        """
        return detector_response(self).ring_up_time(component)

    @property
    def response_model(self):
        return detector_response(self)


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
    name     = 'MWB-EFR',
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
    name     = 'MWB-DMR',
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
        self.f_FSR = 1000 # C_LIGHT / (2.0 * self.arm_length)   # [Hz]

    def ring_up_time(self, component=None) -> float:
        """Nominal DARM or assumed HF-tooth amplitude time, with provenance."""
        return detector_response(self).ring_up_time(component)

    @property
    def response_model(self):
        return detector_response(self)

    @property
    def physical_fsr_hz(self):
        """Physical free spectral range; f_FSR above is a legacy ASD anchor."""
        return C_LIGHT / (2*self.arm_length)


# ── IFO detector catalogue ────────────────────────────────────────────────────

IFO_DETECTORS = {
    'adv_ligo': IFOConfig(
        name       = r'LIGO HF',
        arm_length = 4.0e3,
        asd_anchor = 1.7e-23,
        n_max      = 1000,
        color      = 'black',
        linestyle  = '--',
    ),

    'ligo_data': IFOConfig(
        name       = r'LIGO',
        arm_length = 4.0e3,   # not used now
        asd_anchor = 1.0,     # dummy
        n_max      = 1,
        color      = 'black',
        linestyle  = '-',
    ),
}


# ── IFO noise PSD functions ───────────────────────────────────────────────────

def ifo_noise_psd(freqs: np.ndarray,
                  detector_key: str = 'adv_ligo') -> np.ndarray:

    freqs = np.asarray(freqs, dtype=float)

    if detector_key == 'adv_ligo':
        # OLD model (power law)
        ifo = IFO_DETECTORS[detector_key]
        S_h = np.full_like(freqs, np.nan)
        mask = freqs >= ifo.f_FSR
        S_h[mask] = (ifo.asd_anchor * freqs[mask] / ifo.f_FSR)**2
        return S_h

    elif detector_key == 'ligo_data':
        # NEW: real PSD
        return LIGO_PSD_interp(freqs)

    else:
        raise ValueError(f"Unknown detector key: {detector_key}")


def ifo_asd_envelope(ifo: IFOConfig,
                     freqs: np.ndarray) -> np.ndarray:
    """ASD envelope sqrt(S_h) = A_det * f/f_FSR [Hz^{-1/2}]."""
    return ifo.asd_anchor * (freqs / ifo.f_FSR)


# ═════════════════════════════════════════════════════════════════════════════
# ③ Published detector curves, including those in Shark_fin_plot.zip
#    Muia, Ringwald & Tamarit, HFGWPlotter_Sh (Zenodo:15720342).
#    Source version, units and file hashes: data/provenance.json.
# ═════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=None)
def _load_positive_table(filepath):
    """Load and sort a positive, two-column curve without changing its units."""
    path = Path(filepath)
    data = np.loadtxt(path, delimiter=',' if path.suffix == '.csv' else None)
    if data.ndim != 2 or data.shape[1] != 2 or len(data) < 2:
        raise ValueError(f'{path}: expected at least two rows and two columns')
    if not np.all(np.isfinite(data) & (data > 0)):
        raise ValueError(f'{path}: curve values must be finite and positive')
    data = data[np.argsort(data[:, 0])]
    if np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError(f'{path}: curve coordinates must be unique')
    data.setflags(write=False)
    return data


def _log_interpolate(points, x, y):
    """Power-law interpolation, with NaN outside the tabulated domain."""
    points = np.asarray(points, dtype=float)
    result = np.full_like(points, np.nan)
    mask = np.isfinite(points) & (points >= x[0]) & (points <= x[-1])
    result[mask] = 10.0 ** np.interp(np.log10(points[mask]),
                                    np.log10(x), np.log10(y))
    return result


@dataclass
class TabulatedDetector:
    """Published broadband ASD and an optional, separately supplied binary horizon.

    Noise files contain f [Hz], sqrt(S_h) [Hz^-1/2], except for the explicit
    MAGO unit factors below. Horizon files, when present, contain chirp mass
    [M_sun], d [pc]. Noise-only reference curves have horizon_file=None.
    Binary horizons are not converted to noise, or used as superradiance reach.
    No mechanical or electromagnetic parameters are inferred from these curves.
    """
    name: str
    noise_file: str
    horizon_file: Optional[str]
    color: str
    frequency_scale: float = 1.0
    asd_scale: float = 1.0
    family: ClassVar[str] = 'Tabulated detector'

    @cached_property
    def noise_samples(self):
        """Sorted frequency and ASD samples, converted to Hz and Hz^-1/2."""
        data = _load_positive_table(current_dir / 'data' / 'noise' / self.noise_file)
        return data[:, 0] * self.frequency_scale, data[:, 1] * self.asd_scale

    @property
    def frequency_band(self):
        freqs, _ = self.noise_samples
        return float(freqs[0]), float(freqs[-1])

    def noise_psd(self, freqs):
        """Noise-equivalent strain PSD [Hz^-1], only within the native band."""
        f, asd = self.noise_samples
        return _log_interpolate(freqs, f, asd) ** 2

    def ring_up_time(self, component=None) -> float:
        """Selected component amplitude time [s]; see response_model for scope."""
        return detector_response(self).ring_up_time(component)

    @property
    def response_model(self):
        return detector_response(self)

    @cached_property
    def horizon_samples(self):
        """Original chirp mass [M_sun] and binary distance [pc] samples."""
        if self.horizon_file is None:
            raise ValueError(f'No binary horizon data supplied for {self.name}')
        data = _load_positive_table(current_dir / 'data' / 'shark_fin' / self.horizon_file)
        return data[:, 0], data[:, 1]

    def binary_horizon_pc(self, chirp_mass_solar):
        """Interpolate the supplied binary horizon; NaN outside its mass range."""
        return _log_interpolate(chirp_mass_solar, *self.horizon_samples)


class TabulatedInterferometer(TabulatedDetector):
    family = 'Laser interferometer'


class TabulatedMagneticWeberBar(TabulatedDetector):
    family = 'Magnetic Weber bar'


class LCResonator(TabulatedDetector):
    family = 'Electromagnetic LC resonator'


class RFCavity(TabulatedDetector):
    family = 'Superconducting RF cavity'


class LevitatedSuperconductor(TabulatedDetector):
    family = 'Levitated superconductor'


class DielectricHaloscope(TabulatedDetector):
    family = 'Dielectric haloscope'


class LightShiningThroughWall(TabulatedDetector):
    """Magnetic photon-conversion experiment with single-photon readout.

    Tabulated noise-equivalent strain is suitable for sensitivity comparisons.
    Reach calculations also need photon statistics and directional acceptance;
    these curves alone do not specify a linear matched-filter detector.
    """
    family = 'Light-shining-through-a-wall experiment'


class AxionHelioscope(TabulatedDetector):
    """Magnetic photon conversion with X-ray photon-counting readout.

    The supplied GW sensitivity refers to the receiver's native frequency band
    and incidence assumptions. Photon counts and acceptance enter detection.
    """
    family = 'Axion helioscope'


# Reference curves are distinct from the existing analytic MWB and IFO models.
# In particular, DMRadio-GUT's electromagnetic response is not MWB-DMR.
SHARK_FIN_DETECTORS = {
    'ligo_o4a': TabulatedInterferometer(
        'LIGO O4a (simplified)', 'ligoO4a_simple.csv', 'HorLigoO4as.txt', 'darkorange'),
    'mwb_dmr_broad': TabulatedMagneticWeberBar(
        'Mag. Weber bars (broadband)', 'MWB-DMR-broad.csv', 'HorMWBDMRbroad.txt', 'darkcyan'),
    'lev_sc_30kg': LevitatedSuperconductor(
        'Lev. SC 30 kg', 'SLedDoG_30kg.csv', 'HorLevSup30kg.txt', 'darkturquoise'),
    'mago_broad': RFCavity(
        'MAGO (broadband)', 'MAGO_broad.csv', 'HorMAGObroad.txt', 'rebeccapurple',
        frequency_scale=1 / (2 * np.pi), asd_scale=1 / np.sqrt(2 * np.pi)),
    'holometer': TabulatedInterferometer(
        'Holometer', 'HOLOMETER.txt', 'HorHolometerSingle.txt', 'sienna'),
    'lev_sc_1g': LevitatedSuperconductor(
        'Lev. SC 1 g', 'SLedDoG1g.csv', 'HorLevSup.txt', 'cadetblue'),
    'dmradio_m3_broad': LCResonator(
        'DMRadio-m3 (broadband)', 'DMRadiom3BB.txt', 'HorDMRadio3BB.txt', 'orchid'),
    'dmradio_gut_broad': LCResonator(
        'DMRadio-GUT (broadband)', 'DMRadioGUTBB.txt', 'HorDMRadioGUTBB.txt', 'mediumpurple'),
    'madmax_broad': DielectricHaloscope(
        'MADMAX (broadband)', 'madmax_hybrid_broad_ext.csv', 'HorMadmaxBroad.txt', 'darkmagenta'),
    'dali_ii': DielectricHaloscope(
        'DALI II', 'DALI.txt', 'HorDALI.txt', 'crimson'),
}

# Broadband optical and X-ray reference noise curves; no binary horizons were
# supplied for these five detectors. Do not extend them beyond their table bands.
PHOTON_REGENERATION_DETECTORS = {
    'alps_ii': LightShiningThroughWall(
        'ALPS II', 'ALPSII.txt', None, 'mediumvioletred'),
    'jura': LightShiningThroughWall(
        'JURA', 'JURA.txt', None, 'royalblue'),
    'osqar_ii': LightShiningThroughWall(
        'OSQAR II', 'OSQAR.txt', None, 'darkorange'),
    'cast': AxionHelioscope(
        'CAST', 'CAST.txt', None, 'sienna'),
    'iaxo': AxionHelioscope(
        'IAXO', 'IAXO.txt', None, 'darkgreen'),
}
TABULATED_DETECTORS = {**SHARK_FIN_DETECTORS, **PHOTON_REGENERATION_DETECTORS}

# Unified access for plotting and frequency-domain noise calculations. Existing
# MWB_DETECTORS and IFO_DETECTORS retain their original entries and behaviour.
ALL_DETECTORS = {
    'mwb_efr': ADMX_EFR,
    'mwb_dmr': DMRADIO_GUT,
    **IFO_DETECTORS,
    **TABULATED_DETECTORS,
}

DETECTOR_RESPONSE_MODELS = load_response_models()
PLOTTED_DETECTOR_KEYS = ('mwb_efr', 'mwb_dmr', *IFO_DETECTORS, *SHARK_FIN_DETECTORS)


def detector_response(detector):
    """Component-level physical/assumed timing model for a plotted detector.

    Inspect has_hypothetical_parameters, assumptions and component.scope before
    interpreting a returned time. No claim of a fully calibrated transient
    response is made by attaching a component time to an ASD curve.
    """
    if isinstance(detector, str):
        key, det = detector, ALL_DETECTORS[detector]
    else:
        det = detector
        key = next((k for k, value in ALL_DETECTORS.items() if value is det), None)
        if key is None and isinstance(det, MagneticWeberBar):
            key = 'mwb_efr'  # Same oscillator model, with this instance's values.
    if key not in DETECTOR_RESPONSE_MODELS:
        raise ValueError('No timing model for this detector; excluded photon-conversion curves are not modelled')
    model = DETECTOR_RESPONSE_MODELS[key]
    if isinstance(det, MagneticWeberBar):
        model = model.with_component_parameters('mechanical_mode',
                    frequency_hz=det.f_mech, quality_factor=det.Q_mech)
        model = replace(model, name=det.name or model.name)
        if det.use_LC:
            lc = ResponseComponent('lc_mode', 'oscillator',
                {'frequency_hz': det.f_LC, 'quality_factor': det.Q_EM},
                'configured_parameters', 'Optional resonant LC readout only; not the default broadband curve.',
                'https://arxiv.org/html/2408.01483v2')
            model = replace(model, components=(*model.components, lc))
    return model


def detector_noise_psd(detector, freqs):
    """Return S_h [Hz^-1] for a catalogue key or detector configuration."""
    det = ALL_DETECTORS[detector] if isinstance(detector, str) else detector
    freqs = np.asarray(freqs, dtype=float)
    if isinstance(det, TabulatedDetector):
        return det.noise_psd(freqs)
    if isinstance(det, MagneticWeberBar):
        return mwb_noise_psd(det, freqs)
    if isinstance(det, IFOConfig):
        key = next((k for k, value in IFO_DETECTORS.items() if value is det), None)
        if key is None:
            raise ValueError('IFOConfig must be registered in IFO_DETECTORS')
        return ifo_noise_psd(freqs, key)
    raise TypeError(f'Unsupported detector configuration: {type(det).__name__}')


def plot_detector_response_times(savepath=None, show=True):
    """Compare selected component times; assumptions are labelled explicitly.

    The figure is a timing audit, not a plot of a universal source-duration cut.
    Full component lists, parameters, scopes and sources are in the JSON model.
    """
    with plt.rc_context({'text.usetex': True, 'font.family': 'serif',
                         'font.serif': ['Computer Modern Roman']}):
        fig, ax = plt.subplots(figsize=(12, 8), layout='constrained')
        seen = set()
        for index, key in enumerate(PLOTTED_DETECTOR_KEYS):
            model = detector_response(key)
            component = model.component(model.representative_component)
            assumed = component.origin == 'hypothetical'
            tau = model.ring_up_time()
            if assumed:
                color, marker, label = 'darkorange', 's', 'Hypothetical receiver/component'
            elif model.assumptions:
                color, marker, label = 'teal', '^', 'Nominal/approximate benchmark match'
            else:
                color, marker, label = 'steelblue', 'o', 'Published component benchmark'
            ax.scatter(tau, index, s=55, marker=marker, color=color,
                       label=label if label not in seen else None)
            seen.add(label)
            value = (f'{tau:,.0f} s' if tau >= 1 else
                     f'{tau*1e3:.3g} ms' if tau >= 1e-4 else
                     f'{tau*1e6:.3g} us' if tau >= 1e-6 else f'{tau*1e9:.3g} ns')
            ax.annotate(value, (tau, index), xytext=(7, 0),
                        textcoords='offset points', va='center', fontsize=11, color=color)
        labels=[]
        for key in PLOTTED_DETECTOR_KEYS:
            m=detector_response(key)
            c=m.component(m.representative_component)
            scope={'mechanical_mode':'mechanical resonance', 'darm_pole':'DARM pole',
                   'signal_em_mode':'signal EM mode', 'receiver_rlc':'receiver RLC'}.get(c.name,c.name.replace('_',' '))
            labels.append(f'{m.name} ({scope})')
        ax.set_yticks(range(len(labels)), labels, fontsize=11)
        ax.invert_yaxis()
        ax.set_xscale('log')
        ax.set_xlim(5e-8, 1e5)
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=100))
        ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
        ax.tick_params(direction='in', top=True, right=True, labelsize=11)
        ax.set_xlabel('Component amplitude-envelope time [s]', fontsize=14)
        ax.set_title('Detector response-time benchmarks', fontsize=17)
        ax.grid(axis='x', alpha=0.2)
        ax.legend(loc='lower right', fontsize=11, frameon=False)
        fig.suptitle('Component times apply to the stated modes; they are not minimum source lifetimes.',
                     fontsize=12)
        if savepath is not None:
            path=Path(savepath); path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=200, bbox_inches='tight')
        if show:
            plt.show()
    return fig, ax


# ═════════════════════════════════════════════════════════════════════════════
# ④ Combined noise plot and imported binary-horizon plot
# ═════════════════════════════════════════════════════════════════════════════

def _label_noise_curves(ax):
    """Write each curve's name along its slope, in the same colour as the line."""
    # Frequencies keep labels on distinct, relatively smooth curve segments.
    anchors = {
        'MWB-EFR (analytic)': 5e9,
        'MWB-DMR (analytic)': 5e7,
        'LIGO HF (analytic)': 2e9,
        'LIGO Hanford (existing PSD)': 150,
        'LIGO O4a (simplified)': 220,
        'Mag. Weber bars (broadband)': 1e6,
        'Lev. SC 30 kg': 1e6,
        'MAGO (broadband)': 4e4,
        'Holometer': 5e6,
        'Lev. SC 1 g': 700,
        'DMRadio-m3 (broadband)': 2.5e7,
        'DMRadio-GUT (broadband)': 2e5,
        'MADMAX (broadband)': 8e9,
        'DALI II': 2e10,
        'ALPS II': 5.5e14,
        'JURA': 5.5e14,
        'OSQAR II': 5.5e14,
        'CAST': 7e17,
        'IAXO': 5.5e17,
    }
    if np.log10(ax.get_xlim()[1] / ax.get_xlim()[0]) > 12:
        anchors.update({'MWB-EFR (analytic)': 2e10, 'MAGO (broadband)': 4e4})
    ax.figure.canvas.draw()
    for line in ax.lines:
        name = line.get_label()
        if name.startswith('_'):
            continue
        x, y = line.get_data()
        target = anchors.get(name, np.sqrt(x[0] * x[-1]))
        target = np.clip(target, x[0], x[-1])
        value = 10 ** np.interp(np.log10(target), np.log10(x), np.log10(y))
        # A local log-space chord follows the visible slope, independent of PSD
        # versus ASD units and of the square axes' pixel dimensions.
        sample_x = np.clip(target * np.array([0.85, 1.15]), x[0], x[-1])
        sample_y = 10 ** np.interp(np.log10(sample_x), np.log10(x), np.log10(y))
        p0, p1 = ax.transData.transform(np.column_stack([sample_x, sample_y]))
        angle = np.degrees(np.arctan2(*(p1 - p0)[::-1]))
        offset = -7 if name in ('LIGO O4a (simplified)', 'MWB-DMR (analytic)',
                               'DALI II', 'IAXO', 'Holometer') else 5
        if name in ('LIGO Hanford (existing PSD)', 'LIGO O4a (simplified)',
                    'Mag. Weber bars (broadband)'):
            angle = 0
        text = name.replace('LIGO Hanford (existing PSD)', 'LIGO Hanford\n(existing PSD)')
        # Preserve the point size while fitting full curve names on the compact canvas.
        if ax.figure.get_size_inches()[0] <= 8.5:
            for qualifier in ('broadband', 'analytic', 'simplified'):
                text = text.replace(f' ({qualifier})', f'\n({qualifier})')
        ax.annotate(text, (target, value), xytext=(0, offset),
                    textcoords='offset points', ha='center',
                    va='bottom' if offset >= 0 else 'top',
                    color=line.get_color(), fontsize=12.5, rotation=angle,
                    rotation_mode='anchor', annotation_clip=True,
                    path_effects=[patheffects.withStroke(linewidth=3, foreground='white')])


def plot_all_noise_psds(
        f_min: float = None,
        f_max: float = None,
        n_pts: int = 4000,
        plot_LC: bool = False,
        plot_BBN: bool = False,
        savepath: str = None,
        quantity: str = 'psd',
        show: bool = True,
        include_photon_regeneration: bool = False,
):
    """Plot the selected analytic and published detector noise curves.

    Default quantity is PSD S_h [Hz^-1]; use quantity='asd' for sqrt(S_h).
    Defaults show the original fourteen curves, covering about 10 Hz--97 GHz.
    ALPS II, JURA, OSQAR II, CAST and IAXO are omitted from the default plot;
    include_photon_regeneration=True explicitly restores those reference curves.
    Tables are never extrapolated. Native samples are retained, including narrow
    features that a shared logarithmic grid would otherwise miss. Analytic MWB
    and LIGO HF curves retain their original radiofrequency display range (up to
    about 97 GHz), rather than being extrapolated into the optical/X-ray bands;
    their dashed lines distinguish them from published reference curves.
    Uses the original LaTeX Computer Modern font; pass savepath to export while
    that font context is active. The axes are square with inward ticks and each
    curve is labelled directly, replacing a separate legend.
    """
    if quantity not in ('psd', 'asd'):
        raise ValueError("quantity must be 'psd' or 'asd'")
    if not isinstance(n_pts, (int, np.integer)) or n_pts < 2:
        raise ValueError('n_pts must be an integer of at least 2')
    reference_detectors = (TABULATED_DETECTORS if include_photon_regeneration
                           else SHARK_FIN_DETECTORS)
    bands = [det.frequency_band for det in reference_detectors.values()]
    if f_min is None:
        f_min = min(10.0, *(band[0] for band in bands))
    if f_max is None:
        f_max = max(1e9, *(band[1] for band in bands))
    if not (np.isfinite(f_min) and np.isfinite(f_max) and 0 < f_min < f_max):
        raise ValueError('Frequency limits must satisfy 0 < f_min < f_max')
    freqs = np.geomspace(f_min, f_max, n_pts)
    # Preserve the earlier analytic display domain as the catalogue grows.
    # This is a plotting cap, not a newly asserted physical validity limit.
    analytic_f_max = min(f_max, max(1e9, *(det.frequency_band[1]
                                         for det in SHARK_FIN_DETECTORS.values())))
    analytic_freqs = np.unique(np.r_[freqs[freqs <= analytic_f_max], analytic_f_max])
    transform = np.sqrt if quantity == 'asd' else np.asarray

    # Match the original Computer Modern typography without changing caller styles.
    with plt.rc_context({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'text.latex.preamble': r'\usepackage{amsmath}',
        'font.size': 15,
        'axes.labelsize': 19,
        'axes.titlesize': 20,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
    }):
        # Keep text at its established point sizes on a smaller square canvas.
        side = (18 if quantity == 'psd' else 16) if include_photon_regeneration else 8
        fig, ax = plt.subplots(figsize=(side, side), layout='constrained')
        ax.set_box_aspect(1)

        def draw(f, psd, **style):
            valid = np.isfinite(psd) & (psd > 0) & (f >= f_min) & (f <= f_max)
            if valid.any():
                ax.loglog(f[valid], transform(psd[valid]), **style)

        for det, color in zip(MWB_DETECTORS, ('steelblue', 'teal')):
            f = np.unique(np.r_[analytic_freqs, det.f_mech])
            draw(f, mwb_noise_psd(det, f), color=color, ls='--', lw=1.4,
                 label=f'{det.name} (analytic)', alpha=0.85)
            if plot_LC and det.use_LC:
                draw(f, mwb_noise_psd_LC(det, f), color=color, ls=':', lw=1.3,
                     label=f'{det.name} (LC model)')

        for key, ifo in IFO_DETECTORS.items():
            native = LIGO_PSD_interp.x if key == 'ligo_data' else [ifo.f_FSR]
            grid = analytic_freqs if key == 'adv_ligo' else freqs
            f = np.unique(np.r_[grid, native])
            name = 'LIGO Hanford (existing PSD)' if key == 'ligo_data' else 'LIGO HF (analytic)'
            draw(f, ifo_noise_psd(f, key), color='0.55' if key == 'ligo_data' else '0.2',
                 ls=':' if key == 'ligo_data' else '--', lw=1.2, label=name)

        for det in reference_detectors.values():
            native, _ = det.noise_samples
            low, high = det.frequency_band
            f = np.unique(np.r_[native, freqs[(freqs >= low) & (freqs <= high)]])
            draw(f, det.noise_psd(f), color=det.color, lw=1.7, label=det.name)

        if plot_BBN:
            H0 = 67.4e3 / 3.086e22
            psd = (3 * H0**2 / (2 * np.pi**2)) * 1.8e-6 / freqs**3
            draw(freqs, psd, color='darkgreen', ls=':', lw=1.2, label='BBN bound')

        ax.set_xlabel(r'$f\ [\mathrm{Hz}]$')
        ax.set_ylabel(r'Noise PSD $S_h^{\mathrm{noise}}\ [\mathrm{Hz}^{-1}]$'
                      if quantity == 'psd' else
                      r'$\left(S_h^{\mathrm{noise}}\right)^{1/2}\ [\mathrm{Hz}^{-1/2}]$')
        ax.set_xlim(f_min, f_max)
        # Explicit locators prevent Matplotlib from skipping alternate decades.
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(LogLocator(base=10, subs=(1,), numticks=100))
            axis.set_major_formatter(LogFormatterMathtext(base=10, labelOnlyBase=True))
            axis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
            axis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis='both', which='major', direction='in', top=True,
                       right=True, length=8, width=1.2, labelsize=15)
        ax.tick_params(axis='both', which='minor', direction='in', top=True,
                       right=True, length=4, width=0.9)
        ax.grid(True, which='major', alpha=0.2)
        _label_noise_curves(ax)
        if savepath is not None:
            path = Path(savepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=200, bbox_inches='tight')
        if show:
            plt.show()
    return fig, ax


def plot_shark_fin_horizons(savepath=None, show=True):
    """Plot the ten imported binary-distance curves in their original units.

    These horizons describe binary inspirals, not superradiance signals.
    Event-rate boundaries are copied from the supplied notebook, including its
    conversion from kpc to pc. Display limits match the supplied shark-fin plot.
    """
    with plt.rc_context({'text.usetex': False, 'font.family': 'DejaVu Sans'}):
        fig, ax = plt.subplots(figsize=(5.5, 4), layout='constrained')
        for det in SHARK_FIN_DETECTORS.values():
            ax.loglog(*det.horizon_samples, color=det.color, lw=1.6, label=det.name)
        for fraction, linestyle in (('1', '-'), ('1em3', '--')):
            data = _load_positive_table(
                current_dir / 'data' / 'shark_fin' / f'dyr_fPBH={fraction}_STDRate.txt')
            mass, distance = data[:, 0], data[:, 1] * 1000
            label = '1 event / yr, 100% DM' if fraction == '1' else '1 event / yr, 0.1% DM'
            ax.loglog(mass, distance, color='black', ls=linestyle, lw=1.3, label=label)
            if fraction == '1':
                ax.fill_between(mass, distance, 1e9, color='0.7', alpha=0.4)
        ax.set(xlim=(1e-14, 1), ylim=(1e-8, 10**7.5),
               xlabel=r'Chirp mass $M_c$ [$M_\odot$]', ylabel='Binary distance [pc]',
               title='Imported binary distance reaches')
        ax.grid(True, which='major', alpha=0.2)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=3,
                  fontsize=8, frameon=False)
        if savepath is not None:
            path = Path(savepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=200, bbox_inches='tight')
        if show:
            plt.show()
    return fig, ax


if __name__ == '__main__':
    # Paths resolve beside this module regardless of the launch directory.
    output_dir = current_dir / 'Plots'
    print(f'Saving detector plots to: {output_dir}', flush=True)
    for quantity, basename in (('psd', 'detector_noise_psds'),
                               ('asd', 'detector_noise_asds')):
        for extension in ('pdf', 'png'):
            # Export inside the function's LaTeX font context for both formats.
            fig, _ = plot_all_noise_psds(
                quantity=quantity, show=False,
                savepath=output_dir / f'{basename}.{extension}')
            plt.close(fig)
            print(f'Saved {basename}.{extension}', flush=True)
    fig, _ = plot_shark_fin_horizons(show=False)
    for extension in ('pdf', 'png'):
        fig.savefig(output_dir / f'shark_fin_horizons.{extension}', dpi=200, bbox_inches='tight')
        print(f'Saved shark_fin_horizons.{extension}', flush=True)
    plt.close(fig)
    for extension in ('pdf', 'png'):
        fig, _ = plot_detector_response_times(
            savepath=output_dir / f'detector_response_times.{extension}', show=False)
        plt.close(fig)
        print(f'Saved detector_response_times.{extension}', flush=True)
