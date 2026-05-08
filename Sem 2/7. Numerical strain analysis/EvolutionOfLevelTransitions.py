import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy.integrate import solve_ivp
from pathlib import Path
import re
import pickle


# Use ParamCalculator from Sem 2/0. Scripts from Sem 1
current_dir = Path(__file__).resolve().parent
sem2_dir = current_dir.parent
param_dir = current_dir.parent / "0. Scripts from Sem 1"
sys.path.insert(0, str(param_dir))

# Use CF SR rate utilities from Sem 2/2. Relativistic Superradiance Rate
relativistic_dir = sem2_dir / "2. Relativistic Superradiance Rate"
sys.path.insert(0, str(relativistic_dir))

# Import the required functions from ParamCalculator
from ParamCalculator import (
    calc_superradiance_rate,
    calc_omega_transition,
    calc_transition_rate,
    calc_rg_from_bh_mass,
    G_N
)
from ConvertedFunctions import diff_power_trans_dict
from SuperradianceRateCF import sr_rate_dimensioned
from leaver_superradiance import hydrogen_gamma

available_transitions = list(diff_power_trans_dict.keys())
SOLAR_MASS = 1.988e30  # [kg]



# ---------------------------------------------------------------
#  UTILITY FUNCTIONS
# ---------------------------------------------------------------

def quantum_numbers_to_spectroscopic(n, l):
    """
    Convert quantum numbers to spectroscopic notation (e.g., 5g, 6h).
    
    Args:
        n: principal quantum number
        l: orbital angular momentum quantum number
        
    Returns:
        String in spectroscopic notation (e.g., "5g")
    """
    # Spectroscopic notation letters for l values
    l_to_letter = {
        0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g', 5: 'h',
        6: 'i', 7: 'k', 8: 'l', 9: 'm', 10: 'n', 11: 'o', 12: 'q'
    }
    
    if l not in l_to_letter:
        return f"{n}(l={l})"  # Fallback for l > 12
    
    return f"{n}{l_to_letter[l]}"


# ---------------------------------------------------------------
#  LOGARITHMIC DIFFERENTIAL EQUATIONS
# ---------------------------------------------------------------

MAX_EXP = 50
def rhs_log(t, y, gamma_g, gamma_e, transition_rate):
    """Improved logarithmic RHS function that actually uses log space sensibly"""
    x, y_ = y

    # diff = y_ - x
    # if diff < -MAX_EXP or diff > MAX_EXP:
    #     print(f"[CLIP] Clipper used in rhs_log t={t}, y-x={diff}")
    # # else:
    # #     print(f"[NOT CLIPPED] Clipper not used in rhs_log t={t}, y-x={diff}")

    # diff = np.clip(diff, -MAX_EXP, MAX_EXP)

    # exp_y_minus_x = np.exp(diff)
    # exp_x_minus_y = np.exp(-diff)

    dx = gamma_g + transition_rate * np.exp(y_)
    dy = gamma_e - transition_rate * np.exp(x)
    # print(f"x: {x}, y: {y_}")

    return [dx, dy]


def calc_h_peak_and_fwhm(times, log_h):
    """Return (h_peak, t_peak, fwhm_years) for the strain time series."""
    t = np.asarray(times, dtype=float)
    hs = np.asarray(log_h, dtype=float)

    valid = np.isfinite(t) & np.isfinite(hs)
    if not np.any(valid):
        return np.nan, np.nan, np.nan

    t = t[valid]
    hs = hs[valid]
    if t.size < 3:
        return np.nan, np.nan, np.nan

    peak_idx = int(np.argmax(hs))
    h_peak = float(hs[peak_idx])
    t_peak = float(t[peak_idx])

    if not np.isfinite(h_peak):
        return h_peak, t_peak, np.nan

    log_half = h_peak + np.log10(0.5)

    t_left = None
    for i in range(peak_idx, 0, -1):
        y0, y1 = hs[i - 1], hs[i]
        if (y0 - log_half) * (y1 - log_half) <= 0:
            if y1 == y0:
                t_left = float(t[i])
            else:
                frac = (log_half - y0) / (y1 - y0)
                t_left = float(t[i - 1] + frac * (t[i] - t[i - 1]))
            break

    t_right = None
    for i in range(peak_idx, hs.size - 1):
        y0, y1 = hs[i], hs[i + 1]
        if (y0 - log_half) * (y1 - log_half) <= 0:
            if y1 == y0:
                t_right = float(t[i])
            else:
                frac = (log_half - y0) / (y1 - y0)
                t_right = float(t[i] + frac * (t[i + 1] - t[i]))
            break

    if t_left is None or t_right is None:
        return h_peak, t_peak, np.nan

    return h_peak, t_peak, float(t_right - t_left)


def calc_h_peak_fwhm_bounds(times, log_h):
    """Return (h_peak, t_peak, fwhm_years, t_left, t_right) for the strain time series."""
    t = np.asarray(times, dtype=float)
    hs = np.asarray(log_h, dtype=float)

    valid = np.isfinite(t) & np.isfinite(hs)
    if not np.any(valid):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    t = t[valid]
    hs = hs[valid]
    if t.size < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    peak_idx = int(np.argmax(hs))
    h_peak = float(hs[peak_idx])
    t_peak = float(t[peak_idx])

    if not np.isfinite(h_peak):
        return h_peak, t_peak, np.nan, np.nan, np.nan

    log_half = h_peak + np.log10(0.5)

    t_left = None
    for i in range(peak_idx, 0, -1):
        y0, y1 = hs[i - 1], hs[i]
        if (y0 - log_half) * (y1 - log_half) <= 0:
            if y1 == y0:
                t_left = float(t[i])
            else:
                frac = (log_half - y0) / (y1 - y0)
                t_left = float(t[i - 1] + frac * (t[i] - t[i - 1]))
            break

    t_right = None
    for i in range(peak_idx, hs.size - 1):
        y0, y1 = hs[i], hs[i + 1]
        if (y0 - log_half) * (y1 - log_half) <= 0:
            if y1 == y0:
                t_right = float(t[i])
            else:
                frac = (log_half - y0) / (y1 - y0)
                t_right = float(t[i] + frac * (t[i + 1] - t[i]))
            break

    if t_left is None or t_right is None:
        return h_peak, t_peak, np.nan, (t_left if t_left is not None else np.nan), (t_right if t_right is not None else np.nan)

    return h_peak, t_peak, float(t_right - t_left), float(t_left), float(t_right)


def _find_sr_file(n: int, l: int, m: int, bh_spin: float, sr_data_dir: Path) -> Path:
    """Find the SR file for mode (n,l,m) and exact a*, preferring the newest date."""
    spin_text = f"{bh_spin:.3f}"
    spin_dot = spin_text
    spin_underscore = spin_text.replace('.', '_')
    patterns = (
        f"SR_n{n}l{l}m{m}_at{spin_dot}_aMin*_aMax*_*.dat",
        f"SR_n{n}l{l}m{m}_at{spin_underscore}_aMin*_aMax*_*.dat",
    )

    date_pattern = re.compile(r'_(\d{8})\.dat$')
    matches = []
    for pattern in patterns:
        for path in sr_data_dir.glob(pattern):
            match = date_pattern.search(path.name)
            if match is None:
                continue
            matches.append((match.group(1), path))

    if not matches:
        raise FileNotFoundError(
            f"No SR data file found for mode n={n}, l={l}, m={m}, a*={bh_spin} in '{sr_data_dir}'."
        )

    matches.sort(key=lambda item: item[0])
    return matches[-1][1]


# Extract n and l from spectroscopic notation
def parse_level(level_str):
    l_to_number = {
        's': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5,
        'i': 6, 'k': 7, 'l': 8, 'm': 9, 'n': 10, 'o': 11, 'q': 12, 'r': 13
    }
    n = int(level_str[:-1])
    l_letter = level_str[-1]
    l = l_to_number.get(l_letter)
    if l is None:
        raise ValueError(f"Unknown orbital letter: {l_letter}")
    return n, l


def get_time_unit_config(time_unit="years"):
    time_unit = str(time_unit).strip().lower()

    if time_unit in {"year", "years", "yr", "yrs"}:
        return 1.0, "years"

    if time_unit in {"second", "seconds", "sec", "secs", "s"}:
        return 365.25 * 24 * 3600, "seconds"

    raise ValueError("time_unit must be 'years' or 'seconds'.")

# ---------------------------------------------------------------
#  MAIN SIMULATION
# ---------------------------------------------------------------

def run_simulation(bh_mass_sm=1e-11, bh_spin=0.687, alpha=0.1,
                   transition="3p 2p",
                   gamma_g_override=None, gamma_e_override=None,
                   transition_rate_override=None,
                   distance_kpc=10, t_max_years=None, n_points=int(1e6),
                   sr_rate_source='cf',
                   sr_cf_file_e=None, sr_cf_file_g=None,
                   sr_cf_method='cf',
                   verbose=False):
    """
    Run the simulation with logarithmic variable integration.

    Args:
        bh_mass_sm: Black hole mass in solar masses
        bh_spin: a* (dimensionless spin parameter)
        alpha: fine structure constant
        transition: Transition specification like "3p 2p" or "6g 5g"
        gamma_g_override: Override for ground state SR rate (years^-1)
        gamma_e_override: Override for excited state SR rate (years^-1)
        transition_rate_override: Override for transition rate (years^-1)
        distance_kpc: Distance in kiloparsecs
        t_max_years: Maximum simulation time in years
        n_points: Number of time points
        sr_rate_source: 'cf' (default) to use SuperradianceRateCF data,
            'param' to use ParamCalculator analytic rates,
            or 'hydrogen' to use hydrogen_gamma from leaver_superradiance
        sr_cf_file_e: Optional SR file path for excited level (used when sr_rate_source='cf')
        sr_cf_file_g: Optional SR file path for ground level (used when sr_rate_source='cf')
        sr_cf_method: Interpolation column in SR files ('cf' or 'hydro')
    
    """
    
    # Parse transition string (e.g., "3p 2p" or "6g 5g")
    parts = transition.strip().split()
    if len(parts) != 2:
        raise ValueError(f"Transition must be in format 'ne_level ng_level' (e.g., '3p 2p'), got: {transition}")
    
    level_e_str = parts[0]
    level_g_str = parts[1]
    
    
    n_e, l_e = parse_level(level_e_str)
    n_g, l_g = parse_level(level_g_str)
    
    # For superradiance with l=m constraint
    m_e = l_e
    m_g = l_g

    # --- Constants and conversions ---
    kpc_to_meters = 3.085677581e19
    meters_to_ev = 1 / 1.973269804e-7
    r = distance_kpc * kpc_to_meters * meters_to_ev

    # --- Calculate r_g using ParamCalculator ---
    r_g = calc_rg_from_bh_mass(bh_mass_sm)
    axion_mass = alpha / r_g  # [eV]

    inv_ev_to_years = 2.09e-23  # conversion factor
    ev_to_years = (2.09e-23) ** (-1)  # conversion factor

    # --- Convert a* (bh_spin) to a (not dimensionless) ---
    a = bh_spin * r_g # [eV]^-1

    sr_source_used = sr_rate_source
    sr_file_e_used = None
    sr_file_g_used = None

    if sr_rate_source == 'cf':
        sr_data_dir = sem2_dir / "2. Relativistic Superradiance Rate" / "Mathematica" / "Data"
        sr_file_e = Path(sr_cf_file_e) if sr_cf_file_e is not None else _find_sr_file(
            n=n_e, l=l_e, m=m_e, bh_spin=bh_spin, sr_data_dir=sr_data_dir
        )
        sr_file_g = Path(sr_cf_file_g) if sr_cf_file_g is not None else _find_sr_file(
            n=n_g, l=l_g, m=m_g, bh_spin=bh_spin, sr_data_dir=sr_data_dir
        )

        gamma_e = sr_rate_dimensioned(
            alpha_query=alpha,
            bh_mass_solar=bh_mass_sm,
            filepath=str(sr_file_e),
            method=sr_cf_method,
        )['gamma_natural_eV']
        gamma_g = sr_rate_dimensioned(
            alpha_query=alpha,
            bh_mass_solar=bh_mass_sm,
            filepath=str(sr_file_g),
            method=sr_cf_method,
        )['gamma_natural_eV']

        sr_file_e_used = str(sr_file_e)
        sr_file_g_used = str(sr_file_g)
    elif sr_rate_source == 'param':
        gamma_e = calc_superradiance_rate(l=l_e, m=m_e, n=n_e, a_star=bh_spin, r_g=r_g, alpha=alpha)
        gamma_g = calc_superradiance_rate(l=l_g, m=m_g, n=n_g, a_star=bh_spin, r_g=r_g, alpha=alpha)
    elif sr_rate_source == 'hydrogen':
        # hydrogen_gamma returns the NR SR growth rate in units of 1/M (GM=c=1).
        # Convert to physical natural units [eV] using M = r_g [eV^-1].
        gamma_e = hydrogen_gamma(n=n_e, l=l_e, m=m_e, alpha=alpha, at=bh_spin) / r_g
        gamma_g = hydrogen_gamma(n=n_g, l=l_g, m=m_g, alpha=alpha, at=bh_spin) / r_g
    else:
        raise ValueError("sr_rate_source must be 'cf', 'param', or 'hydrogen'.")

    gamma_e *= ev_to_years # Convert from eV to years^-1
    gamma_g *= ev_to_years # Convert from eV to years^-1

    # === set override values if applicable ===
    if gamma_e_override is not None:
        gamma_e = gamma_e_override
    if gamma_g_override is not None:
        gamma_g = gamma_g_override

    if gamma_e <= 0:
        print("[ERROR] gamma_e is <= zero, most likely the superradiance condition isn't met.")
        raise ValueError("gamma_e must be > 0 for this simulation.")
    if gamma_g <= 0:
        print("[ERROR] gamma_g is <= zero, most likely the superradiance condition isn't met.")
        raise ValueError("gamma_g must be > 0 for this simulation.")

    # Calculate transition frequency using ParamCalculator
    omega = calc_omega_transition(r_g, alpha, n_e, n_g)
    
    # --- Transition rate using ParamCalculator ----
    if transition_rate_override is None:
        # Get transition rate in eV
        transition_rate_ev = calc_transition_rate(
            transition=transition,
            alpha=alpha,
            omega=omega,
            G_N=G_N,
            r_g=r_g
        )
        # Convert to years^-1
        transition_rate = transition_rate_ev * ev_to_years
    else:
        transition_rate = transition_rate_override


    # --- Time setup ---
    calculated_t_max = False
    if t_max_years == None:
        t_max_years = 5 * 1/gamma_g * (np.log(gamma_e) - np.log(transition_rate))
        calculate_t_max = True
    times = np.linspace(0, t_max_years, n_points)
    t_span = (0, times[-1])

    if verbose:
        print("Simulation parameters:")
        print(f"BH mass: {bh_mass_sm} solar masses")
        print(f"BH spin: {bh_spin}")
        print(f"Alpha: {alpha}")
        print(f"Gamma_e = {gamma_e} years⁻¹")
        print(f"Gamma_g = {gamma_g} years⁻¹")
        print(f"transition_rate = {transition_rate} years⁻¹")
        print(f"SR rate source: {sr_source_used}")
        if sr_source_used == 'cf':
            print(f"  SR file (excited): {sr_file_e_used}")
            print(f"  SR file (ground) : {sr_file_g_used}")
        print(f"Total simulated time = {times[-1]} years")
        if calculated_t_max:
            print(f"Calculated t_max: {t_max_years}")
        else:
            print(f"User provided t_max: {t_max_years}")

    LOG10E = np.log10(np.e) # Save variable to avoid repeated log calls
    # --- Event: stop if N_g > some val ---
    def stop_if_ng_too_large(t, y, gamma_g, gamma_e, transition_rate):
        x, y_ = y
        return x * LOG10E - 200  # Trigger at N_g = 1e(whatever is being subtracted)
    stop_if_ng_too_large.terminal = True
    stop_if_ng_too_large.direction = 1

    # --- Event: stop if N_e < 1 ---
    def stop_if_ne_too_small(t, y, gamma_g, gamma_e, transition_rate):
        x, y_ = y
        return y_ - np.log(1.0 - 1e-12)  # Trigger when N_e crosses below 1
    stop_if_ne_too_small.terminal = True
    stop_if_ne_too_small.direction = -1

    # --- Initial conditions (log variables) ---
    n_g0, n_e0 = 1.0, 1.0
    y0_log = [np.log(n_g0), np.log(n_e0)]

    # --- Solve ODE system ---
    sol = solve_ivp(
        rhs_log,
        t_span,
        y0_log,
        args=(gamma_g, gamma_e, transition_rate),
        method='Radau',
        t_eval=times,
        rtol=1e-6,
        atol=1e-9,
        events=[stop_if_ng_too_large,
                stop_if_ne_too_small],
    )

    # --- Use solver’s native arrays (prevents shape mismatch) ---
    times = sol.t
    log_num_g = sol.y[0]
    log_num_e = sol.y[1]

    # --- Check for early termination ---
    if sol.t_events[0].size > 0:
        t_stop = sol.t_events[0][0]
        if verbose:
            print(f"\n[WARNING] Integer Overflow: solve_ivp stopped early, N_g too high or N_e less than one at t ≈ {t_stop:.2f} years.\n")
    else:
        t_stop = times[-1]

    if verbose:
        for i, te in enumerate(sol.t_events):
            print(f"Event {i} triggered at times:", te)

    # --- Gravitational-wave strain ---
    transition_rate_ev_calc = transition_rate * inv_ev_to_years
    log_h = 0.5 * (log_num_e + log_num_g) + np.log(np.sqrt(4 * G_N / (r**2 * omega) * transition_rate_ev_calc))

    results = {
        'times': times,
        'log_num_g': log_num_g,
        'log_num_e': log_num_e,
        'log_h': log_h,
        'status': sol['status'],
        'parameters': {
            'bh_mass_sm': bh_mass_sm,
            'bh_spin': bh_spin,
            'alpha': alpha,
            'transition': transition,
            'l_g': l_g, 'm_g': m_g, 'n_g': n_g,
            'l_e': l_e, 'm_e': m_e, 'n_e': n_e,
            'gamma_g': gamma_g,
            'gamma_e': gamma_e,
            'transition_rate': transition_rate,
            'axion_mass': axion_mass,
            'omega': omega,
            'distance_kpc': distance_kpc,
            'sr_rate_source': sr_source_used,
            'sr_cf_file_e': sr_file_e_used,
            'sr_cf_file_g': sr_file_g_used,
        }
    }

    return results

def scan_transitions_and_save(
    output_pickle="transition_peak_data.pkl",
    transitions=[],
    bh_mass_sm=1e-6,
    bh_spin=0.65,
    alpha_over_l=0.1,
    distance_kpc=10,
    t_max_years=None,
    n_points=int(1e5),
    sr_rate_source="hydrogen",
    sr_cf_method="cf",
    verbose=False,
):

    peak_data = {}

    for transition in transitions:
        level_e_str, level_g_str = transition.strip().split()
        n_g, l_g = parse_level(level_g_str)
        alpha = alpha_over_l * l_g

        try:
            results = run_simulation(
                bh_mass_sm=bh_mass_sm,
                bh_spin=bh_spin,
                alpha=alpha,
                transition=transition,
                distance_kpc=distance_kpc,
                t_max_years=t_max_years,
                n_points=n_points,
                sr_rate_source=sr_rate_source,
                sr_cf_method=sr_cf_method,
                verbose=verbose,
            )

            times = results["times"]
            log_h = results["log_h"] * np.log10(np.e)

            h_peak_log10, t_peak, t_fwhm = calc_h_peak_and_fwhm(times, log_h)

            peak_data[transition] = {
                "h_peak_log10": h_peak_log10,
                "h_peak": 10**h_peak_log10 if np.isfinite(h_peak_log10) else np.nan,
                "t_peak_years": t_peak,
                "t_fwhm_years": t_fwhm,
                "alpha": alpha,
                "parameters": results["parameters"],
                "status": results["status"],
                "success": True,
            }

        except Exception as err:
            print(f"[FAILED] {transition}: {err}")

            peak_data[transition] = {
                "h_peak_log10": np.nan,
                "h_peak": np.nan,
                "t_peak_years": np.nan,
                "t_fwhm_years": np.nan,
                "alpha": np.nan,
                "parameters": None,
                "status": None,
                "success": False,
                "error": str(err),
            }

    with open(output_pickle, "wb") as f:
        pickle.dump(peak_data, f)

    print(f"\nSaved peak data to {output_pickle}")

    print("\nInput parameters:")
    print(f"  bh_mass_sm    : {bh_mass_sm}")
    print(f"  bh_spin       : {bh_spin}")
    print(f"  alpha_over_l  : {alpha_over_l}")
    print(f"  distance_kpc  : {distance_kpc}")
    print(f"  t_max_years   : {t_max_years}")
    print(f"  n_points      : {n_points}")
    print(f"  sr_rate_source: {sr_rate_source}")
    print(f"  sr_cf_method  : {sr_cf_method}")
    print(f"  n_transitions : {len(transitions)}")

    header = f"{'Process':<14} {'h_peak':>12} {'h_peak_log10':>14} {'t_peak [yr]':>14} {'t_fwhm [yr]':>14} {'omega [eV]':>12} {'alpha':>10}"
    print("\n" + header)
    print("-" * len(header))
    for transition in transitions:
        row = peak_data.get(transition, {})
        params = row.get("parameters") or {}
        h_peak = row.get("h_peak", np.nan)
        h_peak_log10 = row.get("h_peak_log10", np.nan)
        t_peak = row.get("t_peak_years", np.nan)
        t_fwhm = row.get("t_fwhm_years", np.nan)
        omega = params.get("omega", np.nan)
        alpha = row.get("alpha", np.nan)

        print(
            f"{transition:<14} "
            f"{h_peak:>12.3e} "
            f"{h_peak_log10:>14.3e} "
            f"{t_peak:>14.3e} "
            f"{t_fwhm:>14.3e} "
            f"{omega:>12.3e} "
            f"{alpha:>10.3e}"
        )

    return peak_data

# ---------------------------------------------------------------
#  PLOTTING
# ---------------------------------------------------------------

def plot_results(results, save_filename=None, time_unit="years", save_plot=False, output_filename=None, plots_subfolder="Plots"):
    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}",
        # Add these font size settings:
        "font.size": 14,               # Base font size
        "axes.titlesize": 16,          # Axis title size
        "axes.labelsize": 15,          # Axis label size
        "xtick.labelsize": 14,         # X-tick label size
        "ytick.labelsize": 14,         # Y-tick label size
        "legend.fontsize": 13,         # Legend font size
        "figure.titlesize": 18         # Figure title size (if you add one)
    })

    
    LOG10E = np.log10(np.e) # Save variable to avoid repeated log calls
    times = results['times']
    log_num_g = results['log_num_g'] * LOG10E
    log_num_e = results['log_num_e'] * LOG10E
    log_h = results['log_h'] * LOG10E
    params = results['parameters']
    status = results['status']
    time_scale, time_unit_label = get_time_unit_config(time_unit)
    plot_times = times * time_scale

    # Detect collapse of N_e (minimum)
    # collapse_index = np.argmin(log_num_e)
    # collapse_time = times[collapse_index]
    # timewindow = max(times) / 17  # [years]
    # xlim = (max(0, collapse_time - timewindow), collapse_time + timewindow - 500)
    
    # Print to terminal (no box)
    print("\n" + "="*60)
    print("SIMULATION RESULTS")
    print("="*60)
    # print(f"Excited-state population collapse at ~{collapse_time:.2f} years.")

    status_dict = {-1: "Integration step failed", 0: "t_max reached", 1: "A termination event occured"}
    print(f"Status: {status_dict[status]}")
    
    # Get spectroscopic notation for transitions
    level_e = quantum_numbers_to_spectroscopic(params['n_e'], params['l_e'])
    level_g = quantum_numbers_to_spectroscopic(params['n_g'], params['l_g'])
    print(f"\nTransition: {level_e} → {level_g}")
    print(f"BH mass: {params['bh_mass_sm']} M☉")
    print(f"BH spin: {params['bh_spin']}")
    print(f"Alpha: {params['alpha']}")
    print(f"Axion mass: {params['axion_mass']:.2e} eV")
    print(f"Distance: {params['distance_kpc']} kpc")
    
    # Compare Gamma_e and Gamma_g
    gamma_e = params.get('gamma_e')
    gamma_g = params.get('gamma_g')
    
    if gamma_e is not None and gamma_g is not None:
        print(f"\nSuperradiance rates:")
        print(f"  Gamma_e ({level_e}): {gamma_e:.2e} years⁻¹")
        print(f"  Gamma_g ({level_g}): {gamma_g:.2e} years⁻¹")
        
        if gamma_e > gamma_g:
            comparison_text = "Gamma_e > Gamma_g"
        elif gamma_e < gamma_g:
            comparison_text = "Gamma_e < Gamma_g"
        else:
            comparison_text = "Gamma_e = Gamma_g"
        print(f"\nComparison: {comparison_text}")
    else:
        print("\nWarning: Gamma values missing from parameters")
    
    print(f"\nTransition rate: {params['transition_rate']:.2e} years⁻¹")
    transition_frequency_hz = params['omega'] / 4.135667696e-6  # Convert to GHz
    print(f"Transition frequency: {params['omega']:.2e} eV ({transition_frequency_hz:.2e} GHz)")

    # Strain summary: peak and full width at half maximum.
    h_peak, t_peak, h_fwhm, t_left, t_right = calc_h_peak_fwhm_bounds(times, log_h)
    t_peak_display = t_peak * time_scale
    print(f"\nPeak strain h_max: {h_peak:.3e} (dimensionless) at t = {t_peak_display:.3e} {time_unit_label}")
    if np.isfinite(h_fwhm):
        h_fwhm_display = h_fwhm * time_scale
        print(f"Strain FWHM: {h_fwhm_display:.3e} {time_unit_label}")
    else:
        print("Strain FWHM: not defined in sampled time window")

    print("="*60 + "\n")

    fig, ax1 = plt.subplots(figsize=(6, 6))
    
    # Plot with clear labels for legend
    ax1.plot(plot_times, log_num_g, label=r'$\log_{10}\, N_g$', color='blue')
    ax1.plot(plot_times, log_num_e, label=r'$\log_{10}\, N_e$', color='orange')
    
    # Create second y-axis for strain h
    ax2 = ax1.twinx()
    ax2.plot(plot_times, log_h, label=r'$\log_{10}\, h$', color='black', alpha=0.7)
    
    # Set scales and labels
    # ax1.set_yscale('log')
    ax1.set_ylabel(r'Occupation Number $\log_{10}\, N$')
    ax1.set_xlabel(f"Time [{time_unit_label}]")
    # ax1.set_ylim(1e25, 1e45)
    # ax1.set_xlim(xlim)
    ax1.grid()
    
    ax2.set_ylabel(r'Strain $\log_{10}\, h$', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    # ax2.set_yscale('log')
    # ax2.set_ylim(h_ylim)
    for label in ax2.get_yticklabels()[::2]:
        label.set_visible(False)

    # Draw FWHM dotted line on strain axis and crop x-limits to FWHM +/- half-FWHM
    fwhm_proxy = None
    if np.isfinite(t_left) and np.isfinite(t_right) and np.isfinite(h_peak):
        time_left = t_left * time_scale
        time_right = t_right * time_scale
        half_max = h_peak + np.log10(0.5)
        ax2.hlines(
            half_max,
            time_left,
            time_right,
            colors='black',
            linestyles=':',
            linewidth=2,
            label='_nolegend_',
        )
        from matplotlib.lines import Line2D
        fwhm_proxy = Line2D([
        ], [], color='black', linestyle=':', linewidth=2, label=r'$\tau_{\text{FWHM}}$')

    # Determine x-limits (display units) from FWHM
    x_min = None
    x_max = None
    if np.isfinite(t_left) and np.isfinite(t_right) and np.isfinite(h_fwhm):
        x_min = max(0.0, (t_left - 15.0 * h_fwhm) * time_scale)
        x_max = (t_right + 0.5 * h_fwhm) * time_scale

    # Helper: set tight y-limits from provided values
    def set_tight_ylim(axis, values, pad_fraction=0.08, min_pad=0.25):
        finite_values = np.asarray(values, dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size == 0:
            return

        y_min = float(np.min(finite_values))
        y_max = float(np.max(finite_values))
        span = y_max - y_min
        pad = max(span * pad_fraction, min_pad)
        if span == 0:
            pad = min_pad

        axis.set_ylim(y_min - pad, y_max + pad)

    # Choose data inside x-limits for y-limits; fall back to full data if mask empty
    if x_min is not None and x_max is not None and x_max > x_min:
        mask = (plot_times >= x_min) & (plot_times <= x_max)
        if np.any(mask):
            vals_ax1 = np.concatenate([log_num_g[mask], log_num_e[mask]])
            vals_ax2 = log_h[mask]
        else:
            vals_ax1 = np.concatenate([log_num_g, log_num_e])
            vals_ax2 = log_h
    else:
        vals_ax1 = np.concatenate([log_num_g, log_num_e])
        vals_ax2 = log_h

    set_tight_ylim(ax1, vals_ax1)
    set_tight_ylim(ax2, vals_ax2)

    # Apply x-limits if computed
    if x_min is not None and x_max is not None and x_max > x_min:
        ax1.set_xlim(x_min, x_max)
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if fwhm_proxy is not None:
        lines2 = lines2 + [fwhm_proxy]
        labels2 = labels2 + [fwhm_proxy.get_label()]
    # Force legend to top-right to avoid overlapping the top-left transition label
    ax1.legend(lines1 + lines2, labels1 + labels2,
            loc='lower left', frameon=True)
    
    # Add comparison text as annotation on plot with LaTeX formatting
    # if gamma_e is not None and gamma_g is not None:
    #     if gamma_e > gamma_g:
    #         comp_text = r'$\Gamma_e^{\text{sr}} > \Gamma_g^{\text{sr}}$'
    #     elif gamma_e < gamma_g:
    #         comp_text = r'$\Gamma_g^{\text{sr}} > \Gamma_e^{\text{sr}}$'
    #     else:
    #         comp_text = r'$\Gamma_e^{\text{sr}} = \Gamma_g^{\text{sr}}$'
        
    #     # Place text in upper left corner
    #     ax1.text(0.02, 0.98, comp_text,
    #             transform=ax1.transAxes,
    #             fontsize=14,
    #             verticalalignment='top')
    
    # Annotate transition in top-left corner using LaTeX
    try:
        trans_label = rf"Transition: ${level_e} \rightarrow {level_g}$"
        ax1.text(
            0.02,
            0.98,
            trans_label,
            transform=ax1.transAxes,
            fontsize=14,
            verticalalignment='top',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=0.3),
        )
    except Exception:
        pass

    # Save to a Plots subfolder next to this script if requested
    if save_plot:
        script_dir = Path(__file__).resolve().parent
        output_dir = script_dir / plots_subfolder
        output_dir.mkdir(parents=True, exist_ok=True)

        if output_filename is None and save_filename is None:
            alpha = params.get('alpha', 'unknown')
            bh_mass = params.get('bh_mass_sm', 'unknown')
            spin = params.get('bh_spin', 'unknown')
            transition_safe = params.get('transition', 'unknown').replace(' ', '_')
            file_name = f"plot_{transition_safe}_alpha={alpha}_bhmass={bh_mass}_spin={spin}.pdf"
        elif output_filename is not None:
            file_name = output_filename
        else:
            file_name = save_filename

        # sanitize
        file_name = re.sub(r'[^\w\-.]', '_', file_name)
        output_path = output_dir / file_name

        plt.tight_layout()
        fig.savefig(output_path, format='pdf', bbox_inches='tight')
        print(f"Saved plot to: {output_path}")

    # Show the plot
    plt.show()


# ---------------------------------------------------------------
#  MAIN EXECUTION
# ---------------------------------------------------------------


if __name__ == "__main__":
    # Shared simulation parameters
    alpha = 0.15
    alpha_over_l = 0.15
    bh_spin = 0.65
    bh_mass_sm = 1e-6
    transition = "3p 2p"
    distance_kpc = 1
    t_max = None
    time_unit = "years"

    # SR-rate source options:
    #   'cf'    -> use SuperradianceRateCF data files (default)
    #   'param' -> use legacy ParamCalculator analytic rates
    #   'hydrogen' -> use hydrogen_gamma NR approximation from leaver_superradiance
    sr_rate_source = 'hydrogen'
    sr_cf_method = 'cf'
    sr_cf_file_e = None  # optional explicit path for excited mode SR file
    sr_cf_file_g = None  # optional explicit path for ground mode SR file

    # Run simulation
    results = run_simulation(
        alpha=alpha,
        bh_spin=bh_spin,
        bh_mass_sm=bh_mass_sm,
        transition=transition,
        t_max_years=t_max,
        sr_rate_source=sr_rate_source,
        sr_cf_method=sr_cf_method,
        sr_cf_file_e=sr_cf_file_e,
        sr_cf_file_g=sr_cf_file_g,
    )

    # peak_data = scan_transitions_and_save(
    #     output_pickle=f"tran_peak_data_alpha_over_l_{str(alpha_over_l).replace('.','p')}.pkl",
    #     transitions=available_transitions,
    #     bh_mass_sm=bh_mass_sm,
    #     bh_spin=bh_spin,
    #     alpha_over_l=alpha_over_l,
    #     t_max_years=None,
    #     n_points=int(1e7),
    #     sr_rate_source=sr_rate_source,
    #     distance_kpc=distance_kpc
    # )

    # Plot results
    plot_results(results, time_unit=time_unit, save_plot=True)
