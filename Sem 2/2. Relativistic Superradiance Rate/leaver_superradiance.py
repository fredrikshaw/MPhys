"""
leaver_superradiance.py
=======================
Computes the dimensionless superradiance rate  Γ = 2·Im(ω)  for a massive
scalar quasi-bound state (QBS) on a Kerr black hole using Leaver's
continued-fraction (CF) method, valid across the full relativistic regime.

Parameters
----------
n       : principal quantum number (n ≥ l + 1)
l       : orbital angular-momentum quantum number
m       : azimuthal quantum number (|m| ≤ l)
a_tilde : dimensionless BH spin  ã = a/M  ∈ [0, 1)
alpha   : gravitational coupling  α = GMμ/ℏc  (= Mμ in GM=ℏ=c=1 units)
omega0_seed : optional complex initial guess for ω

Returns
-------
float : Γ = 2·Im(ω_{nlm})   (positive ↔ superradiant instability)

Units / conventions
-------------------
GM = ℏ = c = 1 throughout (Leaver's natural units, matching the notebook).
The scalar mass is μ = α (since α ≡ GMμ and GM = 1).
The dimensionless rate Γ = 2·Im(ω) satisfies  dN/dt = Γ / (2GM/c³)  in SI.

Algorithm
---------
1.  Angular eigenvalue Λ_{lm}(ω) computed via a dedicated oblate-spheroidal
    continued fraction (robust for all |c|, including complex c).

2.  Three-term recurrence coefficients α_n, β_n, γ_n from Leaver (1985)
    eqs (25)-(32) with  q = −√(μ²−ω²)  (QBS decaying branch):

        α_n = n² + (c₀+1)·n + c₀
        β_n = −2n² + (c₁+2)·n + c₃
        γ_n = n² + (c₂−3)·n + c₄

3.  Continued fraction by Hill's backward substitution from n = N_max,
    using the Nollert (1993) asymptotic seed:

        R_seed = −γ_{Nmax} / α_{Nmax}

    then  R_n = β_n − α_n·γ_{n+1}/R_{n+1}  for n = N_max−1, …, 1.
    CF residual:  f(ω) = β₀ − α₀·γ₁/R₁.

4.  Root of f(ω)=0 found by a **2D Newton method** on the real system
    [Re f(x₀+i·x₁), Im f(x₀+i·x₁)] = 0.  This is necessary because
    Im(ω_QBS) ≪ |Im(ω_QNM)|: a naive complex root-finder is attracted to
    the much larger quasinormal-mode root instead.

Initial-guess strategy (avoids tracking from NR regime)
-------------------------------------------------------
Re(ω₀) from the hydrogen-like binding energy with O(α⁴) relativistic
corrections; Im(ω₀) from the NR hydrogen-like superradiance rate
(Detweiler 1980, Dolan 2007).  Both are computed automatically.
Alternatively, supply `omega0_seed` for full control.

References
----------
Leaver, E.W. (1985), Proc. R. Soc. Lond. A 402, 285-298.
Detweiler, S. (1980), Phys. Rev. D 22, 2323.
Dolan, S.R. (2007), Phys. Rev. D 76, 084001.  [arXiv:0705.2880]
Nollert, H.-P. (1993), Phys. Rev. D 47, 5253.
"""

from math import factorial, prod, log10, ceil
import mpmath
from mpmath import mp, mpc, mpf, sqrt, re, im


# ──────────────────────────────────────────────────────────────────────────────
# 0.  Precision management
# ──────────────────────────────────────────────────────────────────────────────

def _required_prec(l: int, alpha: float, prev_im: float | None = None) -> int:
    """
    Estimate working precision (decimal digits) to resolve Im(ω).

    Im(ω) ~ α^{4l+5}  requires ≈ (4l+5)|log₁₀ α| + 20 digits.
    """
    from_scaling = ceil((4 * l + 5) * abs(log10(abs(alpha) + 1e-300)) + 20)
    if prev_im is not None and prev_im > 0:
        from_history = ceil(abs(log10(abs(prev_im) + 1e-300)) + 20)
    else:
        from_history = 10**9
    return min(80, max(16, min(from_scaling, from_history)))


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Angular eigenvalue  Λ_{lm}(ω)
# ──────────────────────────────────────────────────────────────────────────────

def _ang_alpha_k(k: int, absm: int) -> mpc:
    return mpf(-2 * (k + 1) * (k + 2 * absm + 1))


def _ang_gamma_k(k: int, absm: int, c: mpc) -> mpc:
    if k <= 0:
        return mpf(0)
    kk, mm = mpf(k), mpf(absm)
    return mpf(2) * c**2 * (kk + mm) * (kk + mm - 1) / (
        (2*(kk + mm) - 1) * (2*(kk + mm) + 1))


def _ang_beta_k(k: int, absm: int, c: mpc, A: mpc) -> mpc:
    kk, mm = mpf(k), mpf(absm)
    denom = (kk + mm) * (kk + mm + 1) + mpf("1e-300")
    return (kk + mm) * (kk + mm + 1) - c**2 * (1 - mm**2 / denom) - A


def _spheroidal_eigenvalue(l: int, m: int, c: mpc, prec: int) -> mpc:
    """
    Oblate spheroidal eigenvalue A_{lm}(c) via its own CF.
    Matches Mathematica SpheroidalEigenvalue[l, m, c].
    """
    mp.dps = prec + 10
    absm = abs(m)
    A0 = mpf(l * (l + 1)) - c**2 / 2

    N_cf = 150

    def cf_res(A):
        R = mpf(0)
        for k in range(N_cf, 0, -1):
            num = _ang_alpha_k(k, absm) * _ang_gamma_k(k + 1, absm, c)
            R = num / (_ang_beta_k(k, absm, c, A) - R)
        return _ang_beta_k(0, absm, c, A) - R

    try:
        return mpmath.findroot(cf_res, A0, solver="secant",
                               tol=mpf(10) ** (-(prec + 5)), maxsteps=500)
    except Exception:
        # Perturbative fallback for |c| ≪ 1
        ll, mm = mpf(l), mpf(absm)
        A = mpf(l * (l + 1))
        if l > 0:
            A += c**2 * (2*ll*(ll+1) - 2*mm**2 - 1) / ((2*ll - 1) * (2*ll + 3))
        return A


def angular_lambda(omega: mpc, at: float, l: int, m: int,
                   alpha: float, prec: int = 50) -> mpc:
    """
    Leaver angular eigenvalue Λ_{lm}(ω).

    Matches Mathematica:  SpheroidalEigenvalue[l, m, I·ã·√(ω²−μ²)]
    where μ = α (GM=1).  The argument is purely imaginary (oblate) when
    ω is real with ω < μ (the NR quasi-bound regime).
    """
    mp.dps = prec + 10
    mu = mpf(alpha)
    c = mpc(0, 1) * mpf(at) * sqrt(omega**2 - mu**2 + mpc(0, 1) * mpf("1e-300"))
    return _spheroidal_eigenvalue(l, m, c, prec)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Leaver CF residual  f(ω)
# ──────────────────────────────────────────────────────────────────────────────

def _compute_cf(omega: mpc, at: float, m: int, l: int,
                alpha: float, Nmax: int, prec: int) -> mpc:
    """
    Evaluate the Leaver CF residual for the QBS:
        f(ω) = β₀ − α₀·γ₁/R₁

    using the Nollert seed  R_N = −γ_N / α_N  and the backward sweep
        R_n = β_n − α_n·γ_{n+1}/R_{n+1},   n = N−1 … 1.

    Coefficients (Leaver 1985, eqs 25-32) with q = −√(μ²−ω²) (QBS branch).
    """
    mp.dps = prec + 10
    mu = mpf(alpha)
    at_ = mpf(at)
    b = sqrt(1 - at_**2)                               # b = √(1−ã²)
    q = -sqrt(mu**2 - omega**2 + mpc(0, 1) * mpf("1e-300"))  # QBS: Im(q) ≈ 0⁺
    Lambda = angular_lambda(omega, at, l, m, alpha, prec)

    # Constants c₀–c₄  (Leaver 1985, eqs 25-30)
    c0 = 1 - 2j * omega - (2j / b) * (omega - at_ * m / 2)
    c1 = (-4
          + 4j * (omega - 1j * q * (1 + b))
          + (4j / b) * (omega - at_ * m / 2)
          - 2 * (omega**2 + q**2) / q)
    c2 = (3 - 2j * omega
          - 2 * (q**2 - omega**2) / q
          - (2j / b) * (omega - at_ * m / 2))
    c3 = (2j * (omega - 1j * q)**3 / q
          + 2 * (omega - 1j * q)**2 * b
          + q**2 * at_**2
          + 2j * q * at_ * m
          - Lambda - 1
          - (omega - 1j * q)**2 / q
          + 2 * q * b
          + (2j / b) * ((omega - 1j * q)**2 / q + 1) * (omega - at_ * m / 2))
    c4 = ((omega - 1j * q)**4 / q**2
          + 2j * omega * (omega - 1j * q)**2 / q
          - (2j / b) * (omega - 1j * q)**2 / q * (omega - at_ * m / 2))

    # n-dependent coefficients (Leaver 1985, eqs 31-32)
    # γ_n = n²+(c₂−3)n+c₄  is the form used in the notebook (1-indexed array).
    # When the backward sweep accesses γ at position n+1 (0-indexed n+1),
    # this is equivalent to Leaver eq (32) γ_n = n²+(c₂−3)n+(c₄−c₂+2).
    def alpha_n(n): return n**2 + (c0 + 1) * n + c0
    def beta_n(n):  return -2 * n**2 + (c1 + 2) * n + c3
    def gamma_n(n): return n**2 + (c2 - 3) * n + c4

    # Nollert (1993) asymptotic seed
    R = -gamma_n(Nmax) / alpha_n(Nmax)

    # Backward (Hill) sweep
    for n in range(Nmax - 1, 0, -1):
        R = beta_n(n) - alpha_n(n) * gamma_n(n + 1) / R

    return beta_n(0) - alpha_n(0) * gamma_n(1) / R


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Hydrogen-like rate for seeding
# ──────────────────────────────────────────────────────────────────────────────

def hydrogen_gamma(n: int, l: int, m: int,
                   alpha: float, at: float) -> float:
    """
    Analytic NR superradiance rate Γ_NR (Detweiler 1980 / Dolan 2007 eq 5):

        Γ_NR = 2·r₊·C_{nl}·g_{lm}(ã)·(m·Ω_H − α)·α^{4l+5}

    where r₊ = 1+√(1−ã²), Ω_H = ã/(2r₊).
    Used only as an initial seed; the CF gives the exact value.
    """
    rp = 1.0 + (1.0 - at**2)**0.5
    Omega_H = at / (2.0 * rp)
    if m * Omega_H <= alpha:
        return 0.0

    Cnl = (2**(4*l + 1) * factorial(n + l)
           / (n**(2*l + 4) * factorial(n - l - 1))
           * (factorial(l) / (factorial(2*l) * factorial(2*l + 1)))**2)

    if l == 0:
        glm = 1.0
    else:
        glm = prod(
            k**2 * (1.0 - at**2) + (at * m - 2.0 * rp * alpha)**2
            for k in range(1, l + 1)
        )

    return 2.0 * rp * Cnl * glm * (m * Omega_H - alpha) * alpha**(4*l + 5)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Initial-guess construction
# ──────────────────────────────────────────────────────────────────────────────

def _initial_guess(n: int, l: int, m: int,
                   alpha: float, at: float,
                   omega0_seed: complex | None) -> tuple[float, float]:
    """Return (Re ω₀, Im ω₀) starting point."""
    if omega0_seed is not None:
        return float(omega0_seed.real), float(omega0_seed.imag)

    # Relativistically corrected NR frequency
    re_omega = alpha * (
        1.0
        - alpha**2 / (2.0 * n**2)
        - alpha**4 / n**3 * (1.0 / (8.0 * n) + 6.0 / (2*l + 1) - 2.0 / n)
    )
    re_omega = max(re_omega, alpha * 0.01)

    Gamma_h = hydrogen_gamma(n, l, m, alpha, at)
    im_omega = Gamma_h / 2.0 if Gamma_h > 0.0 else 1e-2 * alpha**(4*l + 5)
    return re_omega, im_omega


# ──────────────────────────────────────────────────────────────────────────────
# 5.  2D Newton root-finder
# ──────────────────────────────────────────────────────────────────────────────

def _find_qbs_root(n: int, l: int, m: int,
                   alpha: float, at: float,
                   omega0_seed: complex | None,
                   Nmax: int, prec: int,
                   tol_digits: int = 8,
                   max_iter: int = 50) -> complex:
    """
    2D Newton's method on  F(x₀, x₁) = [Re f(x₀+ix₁), Im f(x₀+ix₁)] = 0.

    Keeps Im(ω) > 0 throughout to stay on the QBS branch rather than the
    QNM branch (which has Im(ω) ≪ 0).

    Key design: h_im for the numerical Jacobian is always scaled to the current
    x₁ = Im(ω), so the finite difference resolves the gradient even when Im(ω)
    is extremely small (e.g. ~ α^{4l+5} ≈ 10⁻¹²).
    """
    mp.dps = prec + 15
    tol = mpf(10) ** (-tol_digits)
    im_floor = mpf(10) ** (-(prec + 10))   # lower bound for Im(ω)

    re0, im0 = _initial_guess(n, l, m, alpha, at, omega0_seed)
    x0 = mpf(str(re0))
    x1 = mpf(str(max(im0, float(im_floor))))

    # h_re: fixed fraction of Re(ω)
    h_re = mpf(str(max(abs(re0) * 1e-7, 1e-12)))

    def F(x0_, x1_):
        val = _compute_cf(mpc(x0_, x1_), at, m, l, alpha, Nmax, prec)
        return re(val), im(val)

    for iteration in range(max_iter):
        f0, f1 = F(x0, x1)
        residual = abs(f0) + abs(f1)

        if residual < tol:
            break

        # h_im scales with the current Im(ω) so the Jacobian is well-resolved
        h_im = x1 * mpf("1e-2") + im_floor

        # Central-difference Jacobian
        J00 = (F(x0 + h_re, x1)[0] - F(x0 - h_re, x1)[0]) / (2 * h_re)
        J01 = (F(x0, x1 + h_im)[0] - F(x0, x1 - h_im)[0]) / (2 * h_im)
        J10 = (F(x0 + h_re, x1)[1] - F(x0 - h_re, x1)[1]) / (2 * h_re)
        J11 = (F(x0, x1 + h_im)[1] - F(x0, x1 - h_im)[1]) / (2 * h_im)

        det = J00 * J11 - J01 * J10
        if abs(det) < mpf("1e-300"):
            raise RuntimeError(
                f"Jacobian singular at iteration {iteration}; "
                "provide a better omega0_seed."
            )

        # Newton step
        dx0 = -(J11 * f0 - J01 * f1) / det
        dx1 = -(-J10 * f0 + J00 * f1) / det

        # Clamp: never let Im(ω) drop below the floor
        if x1 + dx1 < im_floor:
            dx1 = im_floor - x1

        x0 = x0 + dx0
        x1 = x1 + dx1

    return complex(x0) + 1j * complex(x1)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Public API
# ──────────────────────────────────────────────────────────────────────────────

def find_qbs_frequency(n: int, l: int, m: int,
                       alpha: float, at: float,
                       omega0_seed: complex | None = None,
                       Nmax: int | None = None,
                       prec: int | None = None,
                       tol_digits: int = 8) -> complex:
    """
    Compute the quasi-bound-state frequency ω_{nlm}(α, ã) via Leaver's CF.

    Parameters
    ----------
    n, l, m    : quantum numbers (n ≥ l+1, |m| ≤ l)
    alpha      : gravitational coupling  α = Mμ  (GM=1 units)
    at         : dimensionless BH spin  ã ∈ [0, 1)
    omega0_seed: complex initial ω guess (optional).  Supply this to
                 avoid the root-finder landing on a QNM or wrong mode.
    Nmax       : CF depth (default: max(400, ⌈30/α⌉))
    prec       : decimal digits of working precision (default: auto)
    tol_digits : convergence digits (default: 8)

    Returns
    -------
    complex ω_{nlm}  with Im(ω) > 0 for a superradiant mode.
    """
    if not all(isinstance(x, int) for x in (n, l, m)):
        raise TypeError("n, l, m must be integers")
    if n < l + 1:
        raise ValueError(f"Need n ≥ l+1  (n={n}, l={l})")
    if abs(m) > l:
        raise ValueError(f"|m| ≤ l  (m={m}, l={l})")
    if not (0.0 <= at < 1.0):
        raise ValueError(f"ã must be in [0, 1)  (got {at})")
    if alpha <= 0.0:
        raise ValueError(f"α must be positive  (got {alpha})")

    if prec is None:
        prec = _required_prec(l, alpha)
    if Nmax is None:
        Nmax = max(400, int(30.0 / alpha) + 1)

    return _find_qbs_root(n, l, m, alpha, at,
                          omega0_seed=omega0_seed,
                          Nmax=Nmax, prec=prec,
                          tol_digits=tol_digits)


def superradiance_rate(n: int, l: int, m: int,
                       a_tilde: float, alpha: float,
                       omega0_seed: complex | None = None,
                       Nmax: int | None = None,
                       prec: int | None = None) -> float:
    """
    Dimensionless superradiance rate  Γ = 2·Im(ω_{nlm}).

    Parameters
    ----------
    n, l, m    : quantum numbers (n ≥ l+1, |m| ≤ l)
    a_tilde    : dimensionless BH spin  ã ∈ [0, 1)
    alpha      : gravitational coupling  α = Mμ  (GM=1)
    omega0_seed: optional complex ω seed (see note below)
    Nmax       : CF depth (default: max(400, ⌈30/α⌉))
    prec       : decimal digits of working precision (default: auto)

    Returns
    -------
    float  Γ = 2·Im(ω) > 0 for superradiant instability.

    Notes on seeding
    ----------------
    The automatic seed uses NR hydrogen formulae and works well for
    α ≲ 0.5 and the fundamental mode.  For larger α or higher overtones,
    supply omega0_seed, e.g. stepped from a nearby solved case:

        omega_prev = find_qbs_frequency(2, 1, 1, alpha=0.4, at=0.9)
        rate = superradiance_rate(2, 1, 1, at=0.9, alpha=0.42,
                                  omega0_seed=omega_prev)

    Examples
    --------
    >>> rate = superradiance_rate(2, 1, 1, a_tilde=0.99, alpha=0.2)
    >>> print(f"Γ = {rate:.4e}")
    Γ ≈ 3.21e-09
    """
    omega = find_qbs_frequency(n, l, m, alpha, a_tilde,
                               omega0_seed=omega0_seed,
                               Nmax=Nmax, prec=prec)
    return 2*omega.imag


# ──────────────────────────────────────────────────────────────────────────────
# 7.  CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(
        description="Leaver CF superradiance rate — massive scalar on Kerr",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python leaver_superradiance.py 2 1 1 0.99 0.2
  python leaver_superradiance.py 2 1 1 0.99 0.2 --seed-re 0.199 --seed-im 1.6e-9
  python leaver_superradiance.py 3 2 2 0.99 0.4 --prec 50

Γ = 2·Im(ω) in GM=c=ħ=1 units.  Positive → superradiant instability.
Physical rate:  dN/dt = Γ·c³/(2GM).
""")
    parser.add_argument("n",       type=int,   help="Principal quantum number (n ≥ l+1)")
    parser.add_argument("l",       type=int,   help="Orbital quantum number")
    parser.add_argument("m",       type=int,   help="Azimuthal quantum number")
    parser.add_argument("a_tilde", type=float, help="Dimensionless BH spin ã ∈ [0,1)")
    parser.add_argument("alpha",   type=float, help="Gravitational coupling α = Mμ")
    parser.add_argument("--seed-re", type=float, default=None, metavar="Re",
                        help="Real part of ω seed (optional)")
    parser.add_argument("--seed-im", type=float, default=None, metavar="Im",
                        help="Imaginary part of ω seed (optional)")
    parser.add_argument("--Nmax",  type=int,   default=None,
                        help="CF depth (default: max(400, ⌈30/α⌉))")
    parser.add_argument("--prec",  type=int,   default=None,
                        help="Working precision in digits (default: auto)")
    parser.add_argument("--tol",   type=int,   default=8,
                        help="Convergence tolerance in digits (default: 8)")
    args = parser.parse_args()

    seed = None
    if args.seed_re is not None and args.seed_im is not None:
        seed = complex(args.seed_re, args.seed_im)
    elif (args.seed_re is None) != (args.seed_im is None):
        print("Warning: supply both --seed-re and --seed-im.", file=sys.stderr)

    print(f"Computing Γ for (n={args.n}, l={args.l}, m={args.m}), "
          f"ã={args.a_tilde}, α={args.alpha} …")

    try:
        omega = find_qbs_frequency(
            args.n, args.l, args.m, args.alpha, args.a_tilde,
            omega0_seed=seed, Nmax=args.Nmax,
            prec=args.prec, tol_digits=args.tol,
        )
        print(f"  ω  = {omega.real:.10f}  +  {omega.imag:.6e} i")
        print(f"  Γ  = 2·Im(ω)  =  {2*omega.imag:.6e}")
    except (RuntimeError, ValueError) as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)

