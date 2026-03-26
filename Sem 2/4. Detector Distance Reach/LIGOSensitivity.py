"""
LIGO High-Frequency Sensitivity -- Power-Law Extrapolation
===========================================================

Based on Schnabel & Korobko (2024), arXiv:2409.03019v1.

DERIVATION SUMMARY
------------------
The paper establishes that at frequencies f = n * f_FSR (integer multiples
of the free spectral range f_FSR = c / 2L), interferometric GW observatories
retain significant optical sensitivity even well above the audio band.

The ASD normalised to strain is related to the phase ASD by Eq. (9):

    sqrt(S_h(f)) = (c / 2L omega) * sqrt(S_phi(f))             ... (9)

Above a few kHz, photon shot noise dominates, so S_phi is white (flat in f).
Writing S_phi = S_phi^shot = const, the *noise alone* is frequency-independent.

However the *signal response* at the FSR-comb frequencies falls off with n
because, for a sky-averaged, polarisation-averaged source, the peak response
of the antenna pattern scales as ~1/n (stated explicitly in the paper:
"The maximum response is inversely proportional to the order n of the FSR").

Since the ASD is normalised to the signal strength:

    sqrt(S_h^FSR(n)) = sqrt(S_phi^shot) / |R_sky-avg(n * f_FSR)|

and the sky-averaged peak antenna response scales as 1/n, the strain-
normalised ASD at the n-th FSR comb tooth scales as:

    sqrt(S_h(f_n)) ~ n = f_n / f_FSR

i.e. the sensitivity *degrades linearly* with the comb index n (or
equivalently with frequency f_n = n * f_FSR), giving a power law:

    sqrt(S_h(f)) ~ f^(+1)    [at f = n * f_FSR, n >> 1]

Equivalently, for the one-sided PSD:

    S_h(f) ~ f^2

This is the envelope of the dashed extrapolation lines visible in Fig. 3
of the paper.

ANCHOR VALUES  (A_det and f_FSR read from Fig. 3 of Schnabel & Korobko 2024)
-----------------------------------------------------------------------------
The model is:   sqrt(S_h(f)) = A_det * (f / f_FSR)

Detector             L [m]      f_FSR [Hz]   A_det [/sqrt(Hz)]   n_max
Advanced LIGO        4000       37 500        4.00e-23            1000
GEO 600              1200      125 000        2.00e-21           10000
MHz-GW Explorer       100    1 500 000        2.00e-22              63
GHz-GW Explorer         1  150 000 000        1.00e-20            1000

A_det is the ASD at the *first* FSR comb tooth (n=1), read directly off
Fig. 3.  Every subsequent tooth lies on the straight line
sqrt(S_h) = A_det * n in log-log space (slope +1 decade per decade).
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# -----------------------------------------------------------------------------
# Physical constants
# -----------------------------------------------------------------------------

C_LIGHT = 3.0e8   # m/s

# -----------------------------------------------------------------------------
# Interferometer descriptor
# -----------------------------------------------------------------------------

@dataclass
class IFOConfig:
    """
    Minimal description of a laser interferometer for high-frequency
    sensitivity extrapolation via the Schnabel & Korobko (2024) power law.

    Parameters
    ----------
    name        : display label (LaTeX)
    arm_length  : effective arm / resonator length [m]
    asd_anchor  : A_det = sqrt(S_h) at the *first* FSR comb tooth [Hz^{-1/2}]
                  (read from Fig. 3 of Schnabel & Korobko 2024)
    n_max       : highest comb index to display
    color       : matplotlib colour string
    linestyle   : matplotlib linestyle string
    """
    name        : str
    arm_length  : float
    asd_anchor  : float
    n_max       : int
    color       : str
    linestyle   : str = '-'

    def __post_init__(self):
        self.f_FSR = C_LIGHT / (2.0 * self.arm_length)   # [Hz]


# -----------------------------------------------------------------------------
# Detector catalogue  (anchors from Fig. 3 of Schnabel & Korobko 2024)
# -----------------------------------------------------------------------------

    """'geo600': IFOConfig(
        name        = r'GEO\,600 ($L_\mathrm{eff}=1.2\,\mathrm{km}$)',
        arm_length  = 1.2e3,
        asd_anchor  = 2.0e-21,   # Hz^{-1/2} at f_FSR = 125 kHz
        n_max       = 10000,
        color       = 'darkorange',
        linestyle   = '-',
    ),"""


DETECTORS = {
    'adv_ligo': IFOConfig(
        name        = r'LIGO HF',
        arm_length  = 4.0e3,
        asd_anchor  = 4.0e-23,   # Hz^{-1/2} at f_FSR = 37.5 kHz
        n_max       = 1000,
        color       = 'purple',
        linestyle   = '-',
    ),
}


# -----------------------------------------------------------------------------
# Core power-law model
# -----------------------------------------------------------------------------

def fsr_comb_frequencies(ifo):
    """Return f_n = n * f_FSR for n = 1 ... n_max."""
    n = np.arange(1, ifo.n_max + 1)
    return n * ifo.f_FSR


def asd_power_law(ifo, freqs):
    """
    Schnabel & Korobko (2024) power-law ASD envelope:

        sqrt(S_h(f)) = A_det * (f / f_FSR)

    Valid at (and interpolates between) comb teeth f_n = n * f_FSR.

    Parameters
    ----------
    ifo   : IFOConfig instance
    freqs : array of frequencies [Hz]

    Returns
    -------
    ASD : sqrt(S_h) [Hz^{-1/2}], same shape as freqs
    """
    return ifo.asd_anchor * (freqs / ifo.f_FSR)


def asd_at_comb_teeth(ifo):
    """
    Return (f_comb, asd_comb) for the discrete FSR comb frequencies only.

    Returns
    -------
    f_comb   : np.ndarray, shape (n_max,)
    asd_comb : np.ndarray, shape (n_max,)
    """
    f_comb   = fsr_comb_frequencies(ifo)
    asd_comb = asd_power_law(ifo, f_comb)
    return f_comb, asd_comb


# -----------------------------------------------------------------------------
# Utility: print summary table of f_FSR and A_det
# -----------------------------------------------------------------------------

def print_summary():
    """
    Print the f_FSR and A_det (asd_anchor) values for every detector,
    plus the implied ASD at a few representative frequencies.
    """
    sep = '-' * 90
    print(sep)
    print(
        '{:<36} {:>8} {:>13} {:>15} {:>6}'.format(
            'Detector', 'L [m]', 'f_FSR [Hz]', 'A_det [/rtHz]', 'n_max'
        )
    )
    print(sep)
    for key, ifo in DETECTORS.items():
        name_clean = (
            ifo.name
            .replace('$', '').replace('\\', '')
            .replace('{', '').replace('}', '')
        )
        print(
            '{:<36} {:>8.1e} {:>13.4e} {:>15.2e} {:>6d}'.format(
                name_clean,
                ifo.arm_length,
                ifo.f_FSR,
                ifo.asd_anchor,
                ifo.n_max,
            )
        )
    print(sep)

    print('\nImplied ASD at selected frequencies [Hz^{-1/2}]:')
    probe_freqs = [1e5, 1e6, 1e7, 1e8, 1e9]
    header = '{:<36}'.format('Detector') + ''.join(
        '{:>14.0e}'.format(f) for f in probe_freqs
    )
    print(header)
    print('-' * len(header))
    for key, ifo in DETECTORS.items():
        name_clean = (
            ifo.name
            .replace('$', '').replace('\\', '')
            .replace('{', '').replace('}', '')
        )
        row = '{:<36}'.format(name_clean)
        for f in probe_freqs:
            if f >= ifo.f_FSR:
                asd = asd_power_law(ifo, np.array([f]))[0]
                row += '{:>14.2e}'.format(asd)
            else:
                row += '{:>14}'.format('(< f_FSR)')
        print(row)
    print()


# -----------------------------------------------------------------------------
# Plotting -- standalone IFO sensitivity
# -----------------------------------------------------------------------------

def plot_ligo_high_freq_sensitivity(
        detectors     = None,
        f_min         = 1e4,
        f_max         = 1e11,
        plot_teeth    = False,
        plot_envelope = True,
        savepath      = None,
):
    """
    Plot the high-frequency strain ASD for each interferometer,
    showing both the discrete FSR comb teeth and the f^1 envelope derived
    from Schnabel & Korobko (2024).
    """
    if detectors is None:
        detectors = DETECTORS

    plt.rcParams.update({
        'text.usetex'         : True,
        'font.family'         : 'serif',
        'font.serif'          : ['Computer Modern Roman'],
        'text.latex.preamble' : r'\usepackage{amsmath}',
    })

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.subplots_adjust(top=0.88)

    f_grid = np.logspace(np.log10(f_min), np.log10(f_max), 3000)

    for key, ifo in detectors.items():

        # Discrete comb teeth
        if plot_teeth:
            f_c, asd_c = asd_at_comb_teeth(ifo)
            mask = (f_c >= f_min) & (f_c <= f_max)
            if mask.any():
                ax.scatter(
                    f_c[mask], asd_c[mask],
                    color=ifo.color, s=4, zorder=5, alpha=0.6,
                )

        # Continuous power-law envelope
        if plot_envelope:
            f_env = f_grid[(f_grid >= ifo.f_FSR) & (f_grid <= f_max)]
            if len(f_env) > 0:
                asd_env = asd_power_law(ifo, f_env)
                ax.loglog(
                    f_env, asd_env,
                    color=ifo.color,
                    linewidth=1.8,
                    linestyle=ifo.linestyle,
                    label=ifo.name,
                )

    # Slope annotation
    ligo  = detectors['adv_ligo']
    f_ann = 3e6
    asd_at_ann = asd_power_law(ligo, np.array([f_ann]))[0]
    """ax.annotate(
        r'$\sqrt{S_h} \propto f^{+1}$',
        xy=(f_ann, asd_at_ann),
        xytext=(f_ann * 2.5, asd_at_ann * 0.25),
        fontsize=9, color='dimgray',
        arrowprops=dict(arrowstyle='->', color='dimgray', lw=0.8),
    )"""

    # Vertical FSR marker lines
    for key, ifo in detectors.items():
        if f_min <= ifo.f_FSR <= f_max:
            ax.axvline(
                ifo.f_FSR,
                color=ifo.color, linewidth=0.7,
                linestyle=':', alpha=0.45, zorder=1,
            )

    ax.set_xlabel(r'$f\ [\mathrm{Hz}]$', fontsize=13)
    ax.set_ylabel(r'$\sqrt{S_h}\ [\mathrm{Hz}^{-1/2}]$', fontsize=13)
    ax.set_xlim(f_min, f_max)
    ax.set_ylim(1e-25, 1e-14)
    ax.legend(fontsize=9, loc='upper left', frameon=False)
    ax.set_title(
        r'High-frequency GW sensitivity: $\sqrt{S_h} \propto f$ power law'
        '\n'
        r'(Schnabel \& Korobko 2024, arXiv:2409.03019)',
        fontsize=10, pad=8,
    )
    ax.grid(True)

    if savepath is not None:
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
        print('[saved] {}'.format(savepath))

    plt.tight_layout()
    plt.show()
    return fig, ax


# -----------------------------------------------------------------------------
# Plotting -- overlay with MagneticWeberBar detectors
# -----------------------------------------------------------------------------

def plot_with_mwb_overlay(savepath=None):
    """
    Overlay the interferometer power-law envelopes with ADMX-EFR and
    DMRadio-GUT noise floors from MagneticWeberBar.py, for direct
    comparison with the detectors used in MWBDetectorDistance.py.
    """
    try:
        from MagneticWeberBar import (
            noise_equivalent_strain_broadband,
            ADMX_EFR,
            DMRADIO_GUT,
        )
        mwb_available = True
    except ImportError:
        print('[warning] MagneticWeberBar.py not importable -- '
              'plotting interferometers only.')
        mwb_available = False

    plt.rcParams.update({
        'text.usetex'         : True,
        'font.family'         : 'serif',
        'font.serif'          : ['Computer Modern Roman'],
        'text.latex.preamble' : r'\usepackage{amsmath}',
    })

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.subplots_adjust(top=0.88)

    f_min, f_max = 1e4, 1e13
    f_grid = np.logspace(np.log10(f_min), np.log10(f_max), 5000)

    # Interferometer power-law envelopes
    for key, ifo in DETECTORS.items():
        f_env = f_grid[(f_grid >= ifo.f_FSR) & (f_grid <= f_max)]
        if len(f_env) == 0:
            continue
        asd_env = asd_power_law(ifo, f_env)
        ax.loglog(
            f_env, asd_env,
            color=ifo.color, linewidth=1.0,
            linestyle=ifo.linestyle, label=ifo.name,
        )
        f_c, asd_c = asd_at_comb_teeth(ifo)
        mask = (f_c >= f_min) & (f_c <= f_max)
        """if mask.any():
            ax.scatter(f_c[mask], asd_c[mask],
                       color=ifo.color, s=3, zorder=5, alpha=0.5)"""

    # MWB noise floors
    if mwb_available:
        mwb_freqs   = np.logspace(2, 8, 3000)
        S_h_admx    = noise_equivalent_strain_broadband(ADMX_EFR,    mwb_freqs)
        S_h_dmradio = noise_equivalent_strain_broadband(DMRADIO_GUT, mwb_freqs)
        ax.loglog(mwb_freqs, np.sqrt(S_h_admx),
                  color='steelblue', linewidth=1.0, linestyle='-', alpha=0.8,
                  label=r'ADMX-EFR $\sqrt{S_h^\mathrm{noise}}$')
        ax.loglog(mwb_freqs, np.sqrt(S_h_dmradio),
                  color='teal', linewidth=1.0, linestyle='-', alpha=0.8,
                  label=r'DMRadio-GUT $\sqrt{S_h^\mathrm{noise}}$')

    # Slope annotation
    ligo = DETECTORS['adv_ligo']
    f_a  = 3e6
    asd_a = asd_power_law(ligo, np.array([f_a]))[0]
    """ax.annotate(
        r'$\sqrt{S_h} \propto f^{+1}$',
        xy=(f_a, asd_a),
        xytext=(f_a * 3, asd_a * 0.3),
        fontsize=9, color='dimgray',
        arrowprops=dict(arrowstyle='->', color='dimgray', lw=0.8),
    )"""

    ax.set_xlabel(r'$f\ [\mathrm{Hz}]$', fontsize=13)
    ax.set_ylabel(r'$\sqrt{S_h}\ [\mathrm{Hz}^{-1/2}]$', fontsize=13)
    ax.set_xlim(1e1, 1e9)
    ax.set_ylim(1e-25, 1e-17)
    ax.legend(fontsize=8, loc='upper left', frameon=False)
    ax.grid(True)
    

    if savepath is not None:
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
        print('[saved] {}'.format(savepath))

    plt.tight_layout()
    plt.show()
    return fig, ax


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # Overlay with MWB detectors (requires MagneticWeberBar.py on the path)
    plot_with_mwb_overlay(
        savepath='4. Detector Distance Reach/ligo_highfreq_vs_mwb.pdf',
    )