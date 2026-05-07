import sys
import re
import pickle
from pathlib import Path

import numpy as np

current_dir = Path(__file__).resolve().parent
sem2_dir = current_dir.parent
param_dir = current_dir.parent / "0. Scripts from Sem 1"
sys.path.insert(0, str(param_dir))

relativistic_dir = sem2_dir / "2. Relativistic Superradiance Rate"
sys.path.insert(0, str(relativistic_dir))

from ParamCalculator import (
    calc_superradiance_rate,
    calc_annihilation_rate,
    calc_omega_ann,
    calc_rg_from_bh_mass,
    calc_h_peak_ann,
    calc_n_max,
    calc_delta_astar,
    calc_char_t_ann,
    G_N,
)

from ConvertedFunctions import diff_power_ann_dict
from SuperradianceRateCF import sr_rate_dimensioned
from leaver_superradiance import hydrogen_gamma

available_annihilations = list(diff_power_ann_dict.keys())

EV_TO_YEARS_INV = (2.09e-23) ** (-1)   # eV -> yr^-1
EV_INV_TO_YEARS = 2.09e-23             # eV^-1 -> yr


def parse_level(level_str):
    l_to_number = {
        "s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5,
        "i": 6, "k": 7, "l": 8, "m": 9, "n": 10,
        "o": 11, "q": 12, "r": 13, "t": 14, "u": 15,
    }
    n = int(level_str[:-1])
    l = l_to_number[level_str[-1]]
    return n, l


def _find_sr_file(n, l, m, bh_spin, sr_data_dir):
    spin_text = f"{bh_spin:.3f}"
    patterns = (
        f"SR_n{n}l{l}m{m}_at{spin_text}_aMin*_aMax*_*.dat",
        f"SR_n{n}l{l}m{m}_at{spin_text.replace('.', '_')}_aMin*_aMax*_*.dat",
    )

    date_pattern = re.compile(r"_(\d{8})\.dat$")
    matches = []
    for pattern in patterns:
        for path in sr_data_dir.glob(pattern):
            match = date_pattern.search(path.name)
            if match:
                matches.append((match.group(1), path))

    if not matches:
        raise FileNotFoundError(
            f"No SR data file found for n={n}, l={l}, m={m}, a*={bh_spin}"
        )

    matches.sort(key=lambda item: item[0])
    return matches[-1][1]


def calc_sr_rate_years(
    n, l, m, alpha, bh_mass_sm, bh_spin,
    r_g, sr_rate_source="hydrogen", sr_cf_method="cf", sr_cf_file=None
):
    if sr_rate_source == "cf":
        sr_data_dir = (
            sem2_dir / "2. Relativistic Superradiance Rate"
            / "Mathematica" / "Data"
        )
        sr_file = Path(sr_cf_file) if sr_cf_file else _find_sr_file(
            n, l, m, bh_spin, sr_data_dir
        )

        gamma_ev = sr_rate_dimensioned(
            alpha_query=alpha,
            bh_mass_solar=bh_mass_sm,
            filepath=str(sr_file),
            method=sr_cf_method,
        )["gamma_natural_eV"]

        return gamma_ev * EV_TO_YEARS_INV, str(sr_file)

    if sr_rate_source == "param":
        gamma_ev = calc_superradiance_rate(
            l=l, m=m, n=n, a_star=bh_spin, r_g=r_g, alpha=alpha
        )
        return gamma_ev * EV_TO_YEARS_INV, None

    if sr_rate_source == "hydrogen":
        gamma_ev = hydrogen_gamma(
            n=n, l=l, m=m, alpha=alpha, at=bh_spin
        ) / r_g
        return gamma_ev * EV_TO_YEARS_INV, None

    raise ValueError("sr_rate_source must be 'cf', 'param', or 'hydrogen'.")


def run_annihilation_analytic(
    bh_mass_sm=1e-6,
    bh_spin=0.65,
    alpha=0.5,
    annihilation="5g",
    distance_kpc=10,
    delta_a_star=None,
    sr_rate_source="hydrogen",
    sr_cf_method="cf",
    sr_cf_file=None,
):
    n, l = parse_level(annihilation)
    m = l
    print(annihilation)
    print(n)
    print(l)

    kpc_to_meters = 3.085677581e19
    meters_to_ev = 1 / 1.973269804e-7
    r = distance_kpc * kpc_to_meters * meters_to_ev

    r_g = calc_rg_from_bh_mass(bh_mass_sm)
    bh_mass_ev = r_g / G_N
    axion_mass = alpha / r_g

    if delta_a_star is None:
        delta_a_star = calc_delta_astar(bh_spin, r_g, alpha, n, m)

    if delta_a_star <= 0:
        raise ValueError("delta_a_star <= 0, so the superradiant cloud cannot grow.")

    gamma_sr_years, sr_file_used = calc_sr_rate_years(
        n=n, l=l, m=m,
        alpha=alpha,
        bh_mass_sm=bh_mass_sm,
        bh_spin=bh_spin,
        r_g=r_g,
        sr_rate_source=sr_rate_source,
        sr_cf_method=sr_cf_method,
        sr_cf_file=sr_cf_file,
    )

    if gamma_sr_years <= 0:
        raise ValueError("Superradiance rate is <= 0, so the cloud does not grow.")

    omega_ann = calc_omega_ann(r_g, alpha, n)
    ann_rate_ev = calc_annihilation_rate(
        level=annihilation,
        alpha=alpha,
        omega=omega_ann,
        G_N=G_N,
        r_g=r_g,
    )

    n_max = calc_n_max(
        bh_mass=bh_mass_ev,
        delta_a_star=delta_a_star,
        m_quantum_number=m,
    )

    # Basic SR growth: N(t) = exp(Gamma_sr t), with N(0)=1.
    # Peak/saturation time occurs when N(t_peak) = N_max.
    t_peak_years = np.log(n_max) / gamma_sr_years

    h_peak = calc_h_peak_ann(
        ann_rate=ann_rate_ev,
        omega_ann=omega_ann,
        r=r,
        n_max=n_max,
    )

    # Assumes ParamCalculator returns natural time [eV^-1].
    t_fwhm_years = calc_char_t_ann(
        ann_rate=ann_rate_ev,
        n_max=n_max,
    ) * EV_INV_TO_YEARS

    return {
        "t_peak_years": t_peak_years,
        "t_fwhm_years": t_fwhm_years,
        "h_peak": h_peak,
        "h_peak_log10": np.log10(h_peak) if h_peak > 0 else np.nan,
        "alpha": alpha,
        "parameters": {
            "bh_mass_sm": bh_mass_sm,
            "bh_spin": bh_spin,
            "annihilation": annihilation,
            "n": n,
            "l": l,
            "m": m,
            "gamma_sr_years": gamma_sr_years,
            "ann_rate_ev": ann_rate_ev,
            "n_max": n_max,
            "axion_mass": axion_mass,
            "distance_kpc": distance_kpc,
            "delta_a_star": delta_a_star,
            "sr_rate_source": sr_rate_source,
            "sr_cf_file": sr_file_used,
            "omega": omega_ann,
        },
        "success": True,
    }


def scan_annihilations_and_save(
    output_pickle="annihilation_peak_data.pkl",
    annihilations=None,
    bh_mass_sm=1e-6,
    bh_spin=0.65,
    alpha_over_ls=[0.1],
    distance_kpc=10,
    delta_a_star=None,
    sr_rate_source="hydrogen",
    sr_cf_method="cf",
    scan_alpha=False
):
    if annihilations is None:
        annihilations = available_annihilations

    peak_data = {}

    for annihilation in annihilations:
        print(f"\nRunning annihilation: {annihilation}")
        for alpha_over_l in alpha_over_ls:
            print(f"\n Running alpha_over_l: {alpha_over_l}")
            result_key = (
                annihilation
                if len(alpha_over_ls) == 1
                else f"{annihilation} [a_over_l={alpha_over_l:.6g}]"
            )

            try:
                n, l = parse_level(annihilation)
                if l == 0:
                    raise ValueError("l=0 gives m=0; annihilation SR setup expects m=l>0.")

                alpha = alpha_over_l * l
                print(f"Using alpha = {alpha} from alpha_over_l = {alpha_over_l}, l = {l}")

                peak_data[result_key] = run_annihilation_analytic(
                    bh_mass_sm=bh_mass_sm,
                    bh_spin=bh_spin,
                    alpha=alpha,
                    annihilation=annihilation,
                    distance_kpc=distance_kpc,
                    delta_a_star=delta_a_star,
                    sr_rate_source=sr_rate_source,
                    sr_cf_method=sr_cf_method,
                )

            except Exception as err:
                print(f"[FAILED] {annihilation}: {err}")
                peak_data[result_key] = {
                    "t_peak_years": np.nan,
                    "t_fwhm_years": np.nan,
                    "h_peak": np.nan,
                    "h_peak_log10": np.nan,
                    "omega": np.nan,
                    "alpha": np.nan,
                    "parameters": None,
                    "success": False,
                    "error": str(err),
                }
            print(f"h_peak: {peak_data[result_key]['h_peak_log10']}, t_fwhm_years: {peak_data[result_key]['t_fwhm_years']}, t_peak_years: {peak_data[result_key]['t_peak_years']}")

    with open(output_pickle, "wb") as f:
        pickle.dump(peak_data, f)

    print(f"\nSaved annihilation peak data to {output_pickle}")
    return peak_data


if __name__ == "__main__":
    alpha_over_ls =  np.linspace(0.01, 0.2, 20)
    bh_spin = 0.65
    bh_mass_sm = 1e-6

    peak_data = scan_annihilations_and_save(
        output_pickle=f"ann_peak_data_alpha_over_l_{[str(alpha_over_ls).replace('.', 'p')[1:-1] if len(alpha_over_ls) == 1 else 'sweep'][0]}.pkl",
        annihilations=available_annihilations,
        bh_mass_sm=bh_mass_sm,
        bh_spin=bh_spin,
        alpha_over_ls=alpha_over_ls,
        distance_kpc=10,
        delta_a_star=None,
        sr_rate_source="hydrogen",
    )