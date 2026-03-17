import numpy as np
import sys
from pathlib import Path
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


current_dir = Path(__file__).resolve().parent
sem2_dir = None
for p in current_dir.parents:
    if p.name == "Sem 2":
        sem2_dir = p
        break
if sem2_dir is None:
    sem2_dir = current_dir.parent

script_dir = sem2_dir / "0. Scripts from Sem 1"
if not script_dir.exists():
    # fallback: search upward for the folder name
    for p in current_dir.parents:
        candidate = p / "0. Scripts from Sem 1"
        if candidate.exists():
            script_dir = candidate
            break
sys.path.append(str(script_dir.resolve()))

# Debug: Print the resolved script_dir path
print(f"Resolved script_dir: {script_dir.resolve()}")

# Import the script (expects `calc_superradiance_rate` in ParamCalculator)
from ParamCalculator import (
    calc_omega_transition,
    calc_transition_rate,
    calc_rg_from_bh_mass
)


# Locate the "2. Relativistic Superradiance Rate" folder
relativistic_dir = sem2_dir / "2. Relativistic Superradiance Rate"
if not relativistic_dir.exists():
    # fallback: search upward for the folder name
    for p in current_dir.parents:
        candidate = p / "2. Relativistic Superradiance Rate"
        if candidate.exists():
            relativistic_dir = candidate
            break
sys.path.append(str(relativistic_dir.resolve()))

# Debug: Print the resolved relativistic_dir path
print(f"Resolved relativistic_dir: {relativistic_dir.resolve()}")

# Import the script or functions from the file in "2. Relativistic Superradiance Rate"
from SuperradianceRateCF import sr_rate_dimensioned  # Replace with the actual module and function names

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
    plt.savefig('4.Detector Distance Reach/distance_reach.png', dpi=150)
    plt.show()
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Unit conversion constants
# ─────────────────────────────────────────────────────────────────────────────

EV_TO_J      = 1.602176634e-19   # 1 eV in Joules
HBAR_C_M_EV  = (HBAR * C_LIGHT) / EV_TO_J   # hbar*c in m*eV  (~1.973e-7 m*eV)
EV_TO_SI     = EV_TO_J / HBAR               # 1 eV -> rad/s   (~1.519e15 rad/s)
INV_EV_TO_M  = HBAR_C_M_EV                  # 1 eV^-1 -> m    (~1.973e-7 m)


def h_peak(f_t_SI, transition, alpha, M_solar, filepath):
    """
    Peak strain amplitude at unit distance (r = 1 metre).

    Parameters
    ----------
    f_t_SI : float
        Transition frequency in Hz (SI) — already converted before calling
    alpha : float
        Dimensionless gravitational coupling
    M_solar : float
        Black hole mass in solar masses
    filepath : str
        Path to SR data file

    Returns
    -------
    peak_strain : float
        Strain at r = 1 metre [dimensionless * metres]
    """
    # r_g from natural units (eV^-1) to SI (metres)
    r_g_nat = calc_rg_from_bh_mass(M_solar)        # [eV^-1]
    r_g_SI  = r_g_nat * INV_EV_TO_M                # [m]


    # Transition rate from natural units (eV) to SI (s^-1)
    # Note: pass f_t_SI to calc_transition_rate — check what units
    # calc_transition_rate expects for its frequency argument
    Gamma_t_nat = calc_transition_rate(transition, alpha, f_t_SI, G_NEWTON, r_g_SI)  # [eV]
    Gamma_t_SI  = Gamma_t_nat * EV_TO_SI            # [s^-1]

    # Superradiance rate — already in SI from sr_rate_dimensioned
    SRinfo  = sr_rate_dimensioned(alpha, M_solar, filepath=filepath, method='cf')
    Gamma_sr_SI = SRinfo['gamma_SI']                # [s^-1]

    # omega_t in rad/s
    omega_t_SI = 2 * np.pi * f_t_SI                # [rad/s]

    # Strain at unit distance — all quantities now in SI
    # h = sqrt( 4 G_N / omega_t * Gamma_sr^2 / Gamma_t )  [m * dimensionless]
    peak_strain = np.sqrt(
        4 * G_NEWTON / omega_t_SI * Gamma_sr_SI**2 / Gamma_t_SI
    )

    return peak_strain


def tau(f_t_SI, alpha, M_solar, filepath):
    """
    Signal duration tau_t = 1 / Gamma_sr  [s]

    Gamma_sr is already in SI from sr_rate_dimensioned.
    """
    Gamma_sr_SI = sr_rate_dimensioned(
                      alpha, M_solar, filepath=filepath, method='cf'
                  )['gamma_SI']                     # [s^-1]
    return 1.0 / Gamma_sr_SI                        # [s]


if __name__ == '__main__':

    filepath   = "2. Relativistic Superradiance Rate/Mathematica/SR_n4l2m2_at0.990_aMin0.010_aMax1.200_20260317.dat"
    alpha      = 0.1
    transition = '4d 3d'


    # ── Helper ────────────────────────────────────────────────────────────────
    def compute_point(M_solar):
        r_g_nat     = calc_rg_from_bh_mass(M_solar)
        omega_t_nat = calc_omega_transition(r_g_nat, alpha, 4, 3)
        omega_t_SI  = omega_t_nat * EV_TO_SI
        f_t         = omega_t_SI / (2 * np.pi)

        if not (1e2 <= f_t <= 1e8):
            return f_t, np.nan, np.nan, np.nan, np.nan

        try:
            h_unit  = h_peak(f_t, transition, alpha, M_solar, filepath)
        except Exception:
            return f_t, np.nan, np.nan, np.nan, np.nan

        try:
            tau_val = tau(f_t, alpha, M_solar, filepath)
        except Exception:
            return f_t, np.nan, np.nan, np.nan, np.nan

        S_h_noise = noise_equivalent_strain_broadband(
                        ADMX_EFR, np.array([f_t])
                    )[0]

        d_max_m   = h_unit * np.sqrt(tau_val / S_h_noise)
        d_max_kpc = d_max_m / 3.086e19

        if not np.isfinite(d_max_kpc) or d_max_kpc <= 0:
            return f_t, np.nan, h_unit, tau_val, S_h_noise

        return f_t, d_max_kpc, h_unit, tau_val, S_h_noise

    # ── Stage 1: coarse sweep ─────────────────────────────────────────────────
    M_coarse = np.logspace(-12, 1, 300)
    f_c, d_c, h_c, tau_c, Sn_c = [], [], [], [], []
    for M in M_coarse:
        f, d, h, tv, sn = compute_point(M)
        f_c.append(f);  d_c.append(d)
        h_c.append(h);  tau_c.append(tv);  Sn_c.append(sn)
        # print(f"M = {M:.4e} Msun | f_t = {f:.4e} Hz")

    f_c    = np.array(f_c);    d_c   = np.array(d_c)
    h_c    = np.array(h_c);    tau_c = np.array(tau_c);   Sn_c = np.array(Sn_c)
    fin_c  = np.isfinite(d_c)

    if not fin_c.any():
        raise RuntimeError("No finite points in coarse sweep.")

    # ── Stage 2: dense sweep around f_mech ───────────────────────────────────
    f_mech   = ADMX_EFR.f_mech
    f_lo     = f_mech / 10.0
    f_hi     = f_mech * 10.0
    mask_res = fin_c & (f_c >= f_lo) & (f_c <= f_hi)

    if mask_res.any():
        M_lo = M_coarse[mask_res].min()
        M_hi = M_coarse[mask_res].max()
    else:
        idx_near = np.argmin(np.abs(f_c[fin_c] - f_mech))
        M_near   = M_coarse[fin_c][idx_near]
        M_lo     = 10**(np.log10(M_near) - 1.5)
        M_hi     = 10**(np.log10(M_near) + 1.5)

    M_dense = np.logspace(np.log10(M_lo) - 0.3,
                          np.log10(M_hi) + 0.3, 800)

    f_d, d_d, h_d, tau_d, Sn_d = [], [], [], [], []
    for M in M_dense:
        f, d, h, tv, sn = compute_point(M)
        f_d.append(f);  d_d.append(d)
        h_d.append(h);  tau_d.append(tv);  Sn_d.append(sn)
        # print(f"M = {M:.4e} Msun | f_t = {f:.4e} Hz")

    f_d    = np.array(f_d);    d_d   = np.array(d_d)
    h_d    = np.array(h_d);    tau_d = np.array(tau_d);   Sn_d = np.array(Sn_d)

    # ── Resonance point by interpolation ─────────────────────────────────────
    from scipy.interpolate import interp1d

    f_fin_c  = f_c[fin_c]
    M_fin_c  = M_coarse[fin_c]
    sort_idx = np.argsort(f_fin_c)
    f_s      = f_fin_c[sort_idx];   M_s = M_fin_c[sort_idx]
    _, uniq  = np.unique(f_s, return_index=True)
    f_u      = f_s[uniq];           M_u = M_s[uniq]

    f_to_logM = interp1d(
        np.log10(f_u), np.log10(M_u),
        kind='linear', bounds_error=False, fill_value=np.nan
    )

    f_res_pt = np.nan
    d_res_pt = np.nan
    M_res_pt = np.nan

    log_M_res = f_to_logM(np.log10(f_mech))
    if np.isfinite(log_M_res):
        M_res_exact = 10.0**log_M_res
        print(f"\nResonant mass: {M_res_exact:.4e} Msun | f_t = f_mech = {f_mech:.4e} Hz")
        f_r, d_r, h_r, tau_r, Sn_r = compute_point(M_res_exact)
        if np.isfinite(d_r):
            f_res_pt = f_r
            d_res_pt = d_r
            M_res_pt = M_res_exact

    # ── Merge and sort ────────────────────────────────────────────────────────
    f_all = np.concatenate([f_c,      f_d])
    d_all = np.concatenate([d_c,      d_d])
    M_all = np.concatenate([M_coarse, M_dense])

    sort_idx = np.argsort(f_all)
    f_all    = f_all[sort_idx]
    d_all    = d_all[sort_idx]
    M_all    = M_all[sort_idx]
    fin_all  = np.isfinite(d_all) & np.isfinite(f_all) & (f_all > 0)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "text.usetex"         : True,
        "font.family"         : "serif",
        "font.serif"          : ["Computer Modern Roman"],
        "text.latex.preamble" : r"\usepackage{amsmath}"
    })

    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(top=0.80)

    # Distance reach curve
    ax1.loglog(f_all[fin_all], d_all[fin_all],
               color='steelblue', linewidth=2.0,
               label=fr'$|422\rangle \to |322\rangle,\ \alpha={alpha}$')

    # Resonance point
    if np.isfinite(d_res_pt):
        ax1.scatter([f_res_pt], [d_res_pt],
                    color='firebrick', s=60, zorder=6,
                    label=fr'On resonance ($f_t = f_{{\rm mech}}$)')

        log_M_res_val = np.log10(M_res_pt)
        if abs(log_M_res_val - round(log_M_res_val)) < 0.15:
            M_label = fr'$M = 10^{{{int(round(log_M_res_val))}}}\,M_\odot$'
        else:
            M_label = fr'$M = {M_res_pt:.2e}\,M_\odot$'

        ax1.annotate(
            M_label + '\n' + fr'$f_t = f_{{\rm mech}}$',
            xy         = (f_res_pt, d_res_pt),
            xytext     = (f_res_pt * 6.0, d_res_pt * 0.15),
            fontsize   = 9,
            color      = 'firebrick',
            arrowprops = dict(arrowstyle='->', color='firebrick', lw=0.8),
        )

    # Vertical line at f_mech
    ax1.axvline(f_mech, color='firebrick', linewidth=1.0,
                linestyle=':', alpha=0.7)

    # Right axis: noise curve
    ax_noise    = ax1.twinx()
    freqs_noise = np.logspace(2, 8, 2000)
    S_h         = noise_equivalent_strain_broadband(ADMX_EFR, freqs_noise)
    ax_noise.loglog(freqs_noise, np.sqrt(S_h),
                    color='gray', linewidth=1.2,
                    linestyle='--', alpha=0.5,
                    label=r'$\sqrt{S_h^{\rm noise}}$')
    ax_noise.set_ylabel(
        r'$\left(S_h^{\rm noise}\right)^{1/2}\ [\mathrm{Hz}^{-1/2}]$',
        fontsize=11, color='gray'
    )
    ax_noise.tick_params(axis='y', labelcolor='gray')
    ax_noise.legend(fontsize=9, loc='lower right')

    # Top axis: BH mass
    f_fin = f_all[fin_all];   M_fin = M_all[fin_all]
    sort2 = np.argsort(f_fin)
    f_fin = f_fin[sort2];     M_fin = M_fin[sort2]
    _, uniq2 = np.unique(f_fin, return_index=True)
    f_fin    = f_fin[uniq2];  M_fin = M_fin[uniq2]

    log_M_min   = np.ceil(np.log10(M_fin.min()))
    log_M_max   = np.floor(np.log10(M_fin.max()))
    log_M_ticks = np.arange(log_M_min, log_M_max + 1, 1)

    f_tick_pos    = []
    lM_tick_valid = []
    for lM in log_M_ticks:
        M_t   = 10.0**lM
        idx_t = np.argmin(np.abs(M_fin - M_t))
        f_t_  = f_fin[idx_t]
        xl    = ax1.get_xlim()
        if xl[0] <= f_t_ <= xl[1]:
            f_tick_pos.append(f_t_)
            lM_tick_valid.append(int(lM))

    ax_top = ax1.twiny()
    ax_top.set_xscale('log')
    ax_top.set_xlim(ax1.get_xlim())
    ax_top.set_xticks(f_tick_pos)
    ax_top.set_xticklabels(
        [fr'$10^{{{lM}}}$' for lM in lM_tick_valid],
        fontsize=9
    )
    ax_top.set_xlabel(r'$M_{\rm BH}\ [M_\odot]$', fontsize=12, labelpad=8)

    ax1.set_xlabel(r'$f_t\ [\mathrm{Hz}]$', fontsize=13)
    ax1.set_ylabel(r'$d_{\rm max}\ [\mathrm{kpc}]$', fontsize=13)
    ax1.grid(True, which='both', alpha=0.3)

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax_noise.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2,
               fontsize=10, loc='upper right')

    plt.savefig('4. Detector Distance Reach/distance_reach_mass_sweep.png',
                dpi=150, bbox_inches='tight')
    plt.show()