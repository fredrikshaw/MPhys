"""
superradiance.py
================
Utilities for reading, interpolating, and unit-converting superradiance (SR)
rates computed for a Kerr black hole.

File-naming convention (parsed automatically)
----------------------------------------------
  SR_n{n}l{l}m{m}_at{a*}_aMin{alpha_min}_aMax{alpha_max}_{date}.dat

  e.g. SR_n2l1m1_at0_999_aMin0_010_aMax0_600_20260310.dat

Columns
-------
  alpha       – gravitational fine-structure parameter  α = G_N M_BH μ / ℏ c
                (equivalently  α = r_g μ  in natural units where c = ℏ = 1)
  CF_Gamma    – dimensionless SR rate  Γ^SR r_g  from the continued-fraction
                (CF) method, stored in Mathematica arbitrary-precision format
                e.g.  "1.847...`38.*^-19"
  Hydro_Gamma – same quantity from the hydrodynamic (analytic) approximation,
                stored as a standard double-precision float

Either column can be used for interpolation; the CF column is preferred when
available (it is marked "Failed" for rows where the CF solver did not converge).
The hydrodynamic column is always present but is only a valid approximation in
the low-frequency regime  α/ℓ < 0.1.  This constraint is enforced at runtime:
any attempt to use the hydro data outside this regime raises a ValueError.

Dimensionless → dimensioned conversion
---------------------------------------
  Γ^SR  [s^-1]  =  (Γ^SR r_g)  ×  c / r_g
                =  (Γ^SR r_g)  ×  c³ / (G_N M_BH)

In natural units (c = G_N = ℏ = 1, energies in eV):
  Γ^SR [eV]   =  (Γ^SR r_g)  /  r_g [eV^-1]
  r_g [eV^-1] =  G_N M_BH / (ℏ c)³  in units where ℏ c = 197.3 MeV fm
"""

import os
import re
import numpy as np
from scipy.interpolate import CubicSpline

# ---------------------------------------------------------------------------
# Physical constants (CODATA 2018 / SI)
# ---------------------------------------------------------------------------
_c_SI      = 2.99792458e8          # speed of light              [m s^-1]
_GN_SI     = 6.67430e-11           # Newton's constant            [m^3 kg^-1 s^-2]
_Msun_kg   = 1.98892e30            # solar mass                   [kg]
_hbar_SI   = 1.054571817e-34       # reduced Planck constant      [J s]
_eV_J      = 1.602176634e-19       # 1 eV in Joules               [J]

# The hydrodynamic approximation is only reliable in the low-frequency regime.
# We refuse to use hydro data when  α/ℓ ≥ this threshold.
_HYDRO_MAX_ALPHA_OVER_L = 0.1


def _check_hydro_validity(alpha: float, l: int) -> None:
    """
    Raise ValueError if α/ℓ ≥ _HYDRO_MAX_ALPHA_OVER_L.

    Parameters
    ----------
    alpha : float   – gravitational coupling α
    l     : int     – orbital quantum number ℓ (extracted from the filename)
    """
    ratio = alpha / l
    if ratio >= _HYDRO_MAX_ALPHA_OVER_L:
        raise ValueError(
            f"Hydrodynamic approximation is only valid for α/ℓ < "
            f"{_HYDRO_MAX_ALPHA_OVER_L} (low-frequency regime).  "
            f"Requested α={alpha}, ℓ={l} gives α/ℓ={ratio:.4g}.  "
            "Use method='cf' instead, or reduce α."
        )

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_mathematica_float(s: str) -> float:
    """
    Parse a Mathematica arbitrary-precision number such as
        "1.847621957...`38.*^-19"
    into a Python float.  The backtick and everything after it (precision
    metadata) is stripped; the ``*^`` exponent notation is replaced by ``e``.
    """
    # Strip surrounding whitespace / quotes
    s = s.strip().strip('"')
    # Remove the precision tag  `<digits>.  (and anything before *^)
    s = re.sub(r'`[\d.]+', '', s)
    # Replace Mathematica exponent notation  *^  with Python  e
    s = s.replace('*^', 'e')
    return float(s)


def _parse_metadata_from_filename(filepath: str) -> dict:
    """
    Extract quantum numbers, spin, and alpha range from the filename.

    Returns a dict with keys:
        n, l, m       – integers
        a_star        – float  (dimensionless BH spin  a* = J/(G_N M^2/c))
        alpha_min     – float
        alpha_max     – float
        date          – str  (YYYYMMDD)
    """
    basename = os.path.basename(filepath)
    # Accept either a dot or an underscore as the decimal separator in the
    # numeric fields, since the separator can differ between operating systems
    # and upload tools (e.g. "at0.999" vs "at0_999").
    sep = r'[._]'
    pattern = (
        r'SR_n(\d+)l(\d+)m(\d+)'
        r'_at(\d+)' + sep + r'(\d+)'
        r'_aMin(\d+)' + sep + r'(\d+)'
        r'_aMax(\d+)' + sep + r'(\d+)'
        r'_(\d{8})\.dat'
    )
    m = re.match(pattern, basename)
    if m is None:
        raise ValueError(
            f"Filename '{basename}' does not match the expected pattern\n"
            "  SR_n<n>l<l>m<m>_at<a_int>.<a_dec>"
            "_aMin<amin_int>.<amin_dec>_aMax<amax_int>.<amax_dec>_<YYYYMMDD>.dat\n"
            "(dots or underscores are accepted as the decimal separator)"
        )
    n, l, mq = int(m.group(1)), int(m.group(2)), int(m.group(3))
    a_star    = float(f"{m.group(4)}.{m.group(5)}")
    alpha_min = float(f"{m.group(6)}.{m.group(7)}")
    alpha_max = float(f"{m.group(8)}.{m.group(9)}")
    date      = m.group(10)
    return dict(n=n, l=l, m=mq, a_star=a_star,
                alpha_min=alpha_min, alpha_max=alpha_max, date=date)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_sr_file(filepath: str) -> dict:
    """
    Read an SR data file and return a dict containing the parsed data and
    metadata.  This is a lightweight cached loader – call it once and pass
    the result to the interpolation functions.

    Parameters
    ----------
    filepath : str
        Path to the .dat file.

    Returns
    -------
    data : dict with keys
        'metadata'          – dict from :func:`_parse_metadata_from_filename`
        'alpha'             – np.ndarray, all alpha values in the file
        'cf_gamma'          – np.ndarray, CF Γ^SR r_g  (NaN where "Failed")
        'hydro_gamma'       – np.ndarray, hydrodynamic Γ^SR r_g  (all rows)
        'alpha_cf_valid'    – np.ndarray, alpha values where CF did not fail
        'cf_gamma_valid'    – np.ndarray, CF values at those alpha points
        'alpha_hydro_valid' – np.ndarray, alpha values where α/ℓ < 0.1
        'hydro_gamma_valid' – np.ndarray, hydro values at those alpha points

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If the filename does not match the expected pattern, or if the file
        cannot be parsed.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"SR data file not found: '{filepath}'")

    metadata = _parse_metadata_from_filename(filepath)

    alphas, cf_gammas, hydro_gammas = [], [], []

    with open(filepath, 'r') as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line:
                continue
            # Skip header
            if lineno == 1 and 'alpha' in line.lower():
                continue

            parts = line.split('\t')
            if len(parts) < 3:
                raise ValueError(
                    f"Line {lineno} of '{filepath}' has fewer than 3 tab-separated "
                    f"fields: {line!r}"
                )

            try:
                alpha = _parse_mathematica_float(parts[0])
            except ValueError as exc:
                raise ValueError(
                    f"Cannot parse alpha on line {lineno}: {parts[0]!r}"
                ) from exc

            cf_raw = parts[1].strip().strip('"')
            if 'failed' in cf_raw.lower():
                cf_val = np.nan
            else:
                try:
                    cf_val = _parse_mathematica_float(parts[1])
                except ValueError as exc:
                    raise ValueError(
                        f"Cannot parse CF_Gamma on line {lineno}: {parts[1]!r}"
                    ) from exc

            try:
                hydro_val = float(parts[2].strip().strip('"'))
            except ValueError as exc:
                raise ValueError(
                    f"Cannot parse Hydro_Gamma on line {lineno}: {parts[2]!r}"
                ) from exc

            alphas.append(alpha)
            cf_gammas.append(cf_val)
            hydro_gammas.append(hydro_val)

    alpha       = np.array(alphas,      dtype=np.float64)
    cf_gamma    = np.array(cf_gammas,   dtype=np.float64)  # NaN where Failed
    hydro_gamma = np.array(hydro_gammas, dtype=np.float64)

    cf_valid_mask      = np.isfinite(cf_gamma)
    alpha_cf_valid     = alpha[cf_valid_mask]
    cf_gamma_valid     = cf_gamma[cf_valid_mask]

    # Hydrodynamic data is only reliable for  alpha/l < _HYDRO_MAX_ALPHA_OVER_L
    l = metadata['l']
    hydro_valid_mask   = (alpha / l) < _HYDRO_MAX_ALPHA_OVER_L
    alpha_hydro_valid  = alpha[hydro_valid_mask]
    hydro_gamma_valid  = hydro_gamma[hydro_valid_mask]

    return dict(
        metadata          = metadata,
        alpha             = alpha,
        cf_gamma          = cf_gamma,
        hydro_gamma       = hydro_gamma,
        alpha_cf_valid    = alpha_cf_valid,
        cf_gamma_valid    = cf_gamma_valid,
        alpha_hydro_valid = alpha_hydro_valid,
        hydro_gamma_valid = hydro_gamma_valid,
    )


def interpolate_sr_rate(alpha_query, filepath: str = None, data: dict = None,
                        method: str = 'cf') -> float:
    """
    Return the dimensionless superradiance rate  Γ^SR r_g  at a given α by
    cubic-spline interpolation of the tabulated data.

    Exactly one of *filepath* or *data* must be provided.

    Parameters
    ----------
    alpha_query : float
        The gravitational coupling  α  at which to evaluate the SR rate.
    filepath : str, optional
        Path to the .dat file (used to load data if *data* is not given).
    data : dict, optional
        Pre-loaded data dict as returned by :func:`load_sr_file`.
    method : {'cf', 'hydro'}, optional
        Which column to interpolate.

        ``'cf'`` (default)
            Continued-fraction values.  Rows marked "Failed" are excluded.
            Valid over the full CF-convergent range of the file.

        ``'hydro'``
            Hydrodynamic (analytic) approximation.  **Only permitted when
            α/ℓ < 0.1** (low-frequency regime).  A ValueError is raised if
            *alpha_query* violates this condition.

    Returns
    -------
    float
        Dimensionless SR rate  Γ^SR r_g  at *alpha_query*.  The value is
        returned at machine (float64) precision; no further rounding is applied.

    Raises
    ------
    ValueError
        If *alpha_query* is outside the valid data range; if the hydro method
        is requested but α/ℓ ≥ 0.1; if both or neither of *filepath* / *data*
        are supplied; or if *method* is unrecognised.
    TypeError
        If *alpha_query* is not a real number.
    """
    # --- argument validation ------------------------------------------------
    if (filepath is None) == (data is None):
        raise ValueError(
            "Provide exactly one of 'filepath' or 'data', not both or neither."
        )
    if not np.isreal(alpha_query):
        raise TypeError(f"alpha_query must be a real number, got {type(alpha_query)}")
    alpha_query = float(alpha_query)

    if method not in ('cf', 'hydro'):
        raise ValueError(f"method must be 'cf' or 'hydro', got {method!r}")

    # --- load data if needed ------------------------------------------------
    if data is None:
        data = load_sr_file(filepath)

    l = data['metadata']['l']

    # --- choose column and enforce validity ---------------------------------
    if method == 'cf':
        xs = data['alpha_cf_valid']
        ys = data['cf_gamma_valid']
        col_name = 'CF_Gamma'
    else:
        # Enforce the low-frequency validity condition BEFORE range check so
        # the error message is maximally informative.
        _check_hydro_validity(alpha_query, l)
        xs = data['alpha_hydro_valid']
        ys = data['hydro_gamma_valid']
        col_name = 'Hydro_Gamma (α/ℓ < 0.1 subset)'

    if len(xs) < 2:
        raise ValueError(
            f"Not enough valid data points in column '{col_name}' "
            f"(found {len(xs)}) to perform interpolation."
        )

    alpha_min_valid = float(xs[0])
    alpha_max_valid = float(xs[-1])

    if not (alpha_min_valid <= alpha_query <= alpha_max_valid):
        raise ValueError(
            f"alpha_query={alpha_query} is outside the valid range "
            f"[{alpha_min_valid}, {alpha_max_valid}] for column '{col_name}'."
        )

    # --- cubic spline interpolation -----------------------------------------
    spline = CubicSpline(xs, ys, extrapolate=False)
    result = float(spline(alpha_query))

    if np.isnan(result):
        raise ValueError(
            f"Interpolation returned NaN at alpha={alpha_query}; "
            "check that the data file is not corrupt."
        )

    return result


def sr_rate_dimensioned(alpha_query, bh_mass_solar: float,
                        filepath: str = None, data: dict = None,
                        method: str = 'cf') -> dict:
    """
    Convert the dimensionless SR rate  Γ^SR r_g  to a physical rate for a
    black hole of given mass.

    The conversion is:
        Γ^SR [s^-1]  =  (Γ^SR r_g) × c³ / (G_N M_BH)
        Γ^SR [eV]    =  Γ^SR [s^-1] × ℏ  /  eV

    Parameters
    ----------
    alpha_query : float
        Gravitational coupling  α.
    bh_mass_solar : float
        Black-hole mass in solar masses  (M_BH / M_☉).
    filepath : str, optional
        Path to the .dat file (used if *data* is not provided).
    data : dict, optional
        Pre-loaded data dict from :func:`load_sr_file`.
    method : {'cf', 'hydro'}, optional
        Which column to interpolate (passed to :func:`interpolate_sr_rate`).

    Returns
    -------
    result : dict with keys
        'alpha'               – float, the queried α value
        'bh_mass_solar'       – float, M_BH in solar masses
        'gamma_dimensionless' – float, Γ^SR r_g  (dimensionless)
        'gamma_SI'            – float, Γ^SR in s^-1 (SI)
        'gamma_natural_eV'    – float, Γ^SR in eV  (natural units ℏ=c=1)
        'r_g_m'               – float, gravitational radius r_g = G_N M/c² [m]
        'r_g_inv_eV'          – float, r_g in eV^-1

    Raises
    ------
    ValueError
        If *bh_mass_solar* is not positive, or propagated from
        :func:`interpolate_sr_rate`.
    TypeError
        If *bh_mass_solar* is not a real number.
    """
    # --- argument validation ------------------------------------------------
    if not np.isreal(bh_mass_solar):
        raise TypeError(
            f"bh_mass_solar must be a real number, got {type(bh_mass_solar)}"
        )
    bh_mass_solar = float(bh_mass_solar)
    if bh_mass_solar <= 0:
        raise ValueError(
            f"bh_mass_solar must be positive, got {bh_mass_solar}"
        )

    # --- load data once so interpolation and metadata share the same object -
    if data is None and filepath is not None:
        data = load_sr_file(filepath)
    elif data is None:
        raise ValueError("Provide either 'filepath' or 'data'.")

    # --- dimensionless rate -------------------------------------------------
    gamma_dl = interpolate_sr_rate(
        alpha_query, data=data, method=method
    )

    # --- gravitational radius -----------------------------------------------
    M_kg   = bh_mass_solar * _Msun_kg
    r_g_m  = _GN_SI * M_kg / _c_SI**2          # r_g = G_N M / c²  [m]

    # --- SI rate ------------------------------------------------------------
    # Γ [s^-1] = (Γ r_g) * c / r_g
    gamma_SI = gamma_dl * _c_SI / r_g_m

    # --- natural-units rate (ℏ = c = 1, energy in eV) ----------------------
    # Γ [eV] = Γ [s^-1] * ℏ [J s] / eV_to_J
    gamma_eV = gamma_SI * _hbar_SI / _eV_J

    # gravitational radius in natural units  (r_g [eV^-1] = r_g [m] / (ℏc [m eV]))
    hbar_c_m_eV = _hbar_SI * _c_SI / _eV_J      # ℏc in m·eV
    r_g_inv_eV  = r_g_m / hbar_c_m_eV           # r_g in eV^-1

    return dict(
        alpha               = alpha_query,
        bh_mass_solar       = bh_mass_solar,
        gamma_dimensionless = gamma_dl,
        gamma_SI            = gamma_SI,
        gamma_natural_eV    = gamma_eV,
        r_g_m               = r_g_m,
        r_g_inv_eV          = r_g_inv_eV,
    )


# ---------------------------------------------------------------------------
# Convenience: pretty-print the result dict
# ---------------------------------------------------------------------------

def print_sr_result(result: dict) -> None:
    """Pretty-print the dict returned by :func:`sr_rate_dimensioned`."""
    print(f"  α                    = {result['alpha']:.6g}")
    print(f"  M_BH                 = {result['bh_mass_solar']:.6g} M_☉")
    print(f"  r_g                  = {result['r_g_m']:.6e} m")
    print(f"  r_g                  = {result['r_g_inv_eV']:.6e} eV^-1")
    print(f"  Γ^SR r_g (dimless)   = {result['gamma_dimensionless']:.15e}")
    print(f"  Γ^SR      (SI)       = {result['gamma_SI']:.6e} s^-1")
    print(f"  Γ^SR      (natural)  = {result['gamma_natural_eV']:.6e} eV")


# ---------------------------------------------------------------------------
# Quick self-test when run as a script
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    # Allow overriding the test file path from the command line
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        test_file = test_file = "2. Relativistic Superradiance Rate/Mathematica/SR_n2l1m1_at0.999_aMin0.010_aMax0.600_20260310.dat"

    if not os.path.isfile(test_file):
        print(f"Test file '{test_file}' not found – skipping self-test.")
        sys.exit(0)

    print(f"Loading: {test_file}")
    d = load_sr_file(test_file)
    meta = d['metadata']
    l = meta['l']
    hydro_max_alpha = _HYDRO_MAX_ALPHA_OVER_L * l
    print(f"  n={meta['n']}, l={l}, m={meta['m']}, "
          f"a*={meta['a_star']}, "
          f"alpha in [{meta['alpha_min']}, {meta['alpha_max']}]")
    print(f"  Total rows       : {len(d['alpha'])}")
    print(f"  CF-valid rows    : {len(d['alpha_cf_valid'])}")
    print(f"  Hydro-valid rows : {len(d['alpha_hydro_valid'])}  "
          f"(alpha/l < {_HYDRO_MAX_ALPHA_OVER_L}  =>  alpha < {hydro_max_alpha})")

    # --- CF interpolation ---------------------------------------------------
    alpha_test = 0.3
    print(f"\nInterpolating at alpha = {alpha_test} (CF method):")
    rate = interpolate_sr_rate(alpha_test, data=d)
    print(f"  Gamma^SR r_g = {rate:.15e}")

    # --- Hydro interpolation inside valid regime ----------------------------
    alpha_hydro_test = 0.05   # alpha/l = 0.05 < 0.1  for l=1
    print(f"\nInterpolating at alpha = {alpha_hydro_test} (hydro method, "
          f"alpha/l = {alpha_hydro_test/l:.3g}):")
    rate_h = interpolate_sr_rate(alpha_hydro_test, data=d, method='hydro')
    print(f"  Gamma^SR r_g = {rate_h:.15e}")

    # --- Hydro rejection outside valid regime -------------------------------
    alpha_bad = 0.15   # alpha/l = 0.15 > 0.1  for l=1
    print(f"\nAttempting hydro at alpha = {alpha_bad} "
          f"(alpha/l = {alpha_bad/l:.3g}, should raise ValueError):")
    try:
        interpolate_sr_rate(alpha_bad, data=d, method='hydro')
    except ValueError as exc:
        print(f"  Correctly rejected: {exc}")

    # --- Dimensioned rates --------------------------------------------------
    print(f"\nDimensioned SR rate at alpha = {alpha_test}, M_BH = 1 M_sun:")
    res = sr_rate_dimensioned(alpha_test, bh_mass_solar=1.0, data=d)
    print_sr_result(res)

    print(f"\nDimensioned SR rate at alpha = {alpha_test}, M_BH = 10 M_sun:")
    res10 = sr_rate_dimensioned(alpha_test, bh_mass_solar=10.0, data=d)
    print_sr_result(res10)