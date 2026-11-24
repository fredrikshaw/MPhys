"""Axion potential and analytic curvatures for the MassDist.py conventions."""

import numpy as np
from scipy.linalg import eigvalsh


FREQUENCY = 2 * np.pi
PAIR_BLOCK_SIZE = 2048


def instanton_charges(cy):
    """Return the prime toric divisor charges in the default CY divisor basis."""
    return cy.glsm_charge_matrix(include_origin=False).T


class AxionPotential:
    """Precompute the coefficients of the signed cosines in the potential."""

    def __init__(self, q, tau, g_inv, volume, planck_mass_ev, *, W0=1.0, divisor_volumes=None):
        self._q = np.asarray(q, dtype=float)
        tau = np.asarray(tau, dtype=float)
        g_inv = np.asarray(g_inv, dtype=float)
        if self._q.ndim != 2 or self._q.shape[0] == 0 or self._q.shape[1] == 0:
            raise ValueError("The charge matrix must be nonempty and two-dimensional.")
        n_axions = self._q.shape[1]
        if tau.shape != (n_axions,) or g_inv.shape != (n_axions, n_axions):
            raise ValueError("The moduli and inverse metric must use the charge matrix basis.")
        if not all(np.isfinite(values).all() for values in (self._q, tau, g_inv)):
            raise ValueError("The charges, moduli and inverse metric must be finite.")
        if not np.isfinite(volume) or volume <= 0:
            raise ValueError("The Calabi-Yau volume must be positive and finite.")
        if not np.isfinite(planck_mass_ev) or planck_mass_ev <= 0:
            raise ValueError("The Planck mass must be positive and finite.")
        if not np.isfinite(W0):
            raise ValueError("W0 must be finite.")

        self._log_planck = np.log(planck_mass_ev)
        self._log_prefactor = np.log(8 * np.pi) - 2 * np.log(volume)
        self._canonical_factor = np.linalg.cholesky(g_inv)
        canonical_q = self._q @ self._canonical_factor
        action_volumes = self._q @ tau if divisor_volumes is None else np.asarray(divisor_volumes)
        if action_volumes.shape != (len(self._q),):
            raise ValueError("Expected one divisor volume per instanton charge.")

        self._pair_a, self._pair_b = np.tril_indices(len(self._q), k=-1)
        pair_volumes = action_volumes[self._pair_a] + action_volumes[self._pair_b]
        charge_products = canonical_q @ canonical_q.T
        pair_prefactors = np.pi * charge_products[self._pair_a, self._pair_b] + pair_volumes
        single_prefactors = W0 * action_volumes
        if not all(np.isfinite(values).all() for values in (single_prefactors, pair_prefactors)):
            raise ValueError("Non-finite instanton prefactors.")

        self._single_sign = np.sign(single_prefactors)
        self._pair_sign = np.sign(pair_prefactors)
        # Form logarithms before exponentiating to keep an overall small
        # scale from underflowing before the mass conversion.
        with np.errstate(divide="ignore"):
            self._single_log = np.log(np.abs(single_prefactors)) - FREQUENCY * action_volumes
            self._pair_log = np.log(np.abs(pair_prefactors)) - FREQUENCY * pair_volumes

        # A zero frequency contributes to the potential but never to its
        # Hessian. Keep it out of the curvature's reference amplitude.
        _, charge_ids = np.unique(self._q, axis=0, return_inverse=True)
        self._active_single = np.flatnonzero(np.any(self._q != 0, axis=1) & (self._single_sign != 0))
        self._active_pair = np.flatnonzero(
            (charge_ids[self._pair_a] != charge_ids[self._pair_b]) & (self._pair_sign != 0)
        )

    def _cosines(self, theta, single, pair):
        if theta is None:
            return 1.0, 1.0
        theta = np.asarray(theta, dtype=float).reshape(-1)
        if theta.shape != (self._q.shape[1],):
            raise ValueError("The axion point must have one angle per charge column.")
        phases = FREQUENCY * (self._q @ theta)
        return (
            np.cos(phases[single]),
            np.cos(phases[self._pair_a[pair]] - phases[self._pair_b[pair]]),
        )

    def _weights(self, theta, single, pair):
        single_log = self._single_log[single]
        pair_log = self._pair_log[pair]
        reference = max(single_log.max(initial=-np.inf), pair_log.max(initial=-np.inf))
        if reference == -np.inf:
            return 0.0, np.zeros_like(single_log), np.zeros_like(pair_log)
        single_cos, pair_cos = self._cosines(theta, single, pair)
        with np.errstate(under="ignore"):
            weights = self._single_sign[single] * np.exp(single_log - reference) * single_cos
            pair_weights = self._pair_sign[pair] * np.exp(pair_log - reference) * pair_cos
        return reference, weights, pair_weights

    def __call__(self, theta=None):
        """Evaluate the potential in eV^4 using the script's Planck convention."""
        reference, weights, pair_weights = self._weights(theta, slice(None), slice(None))
        total = weights.sum() + pair_weights.sum()
        if total == 0:
            return 0.0
        log_amplitude = self._log_prefactor + 4 * self._log_planck + reference
        return float(-np.sign(total) * np.exp(np.log(abs(total)) + log_amplitude))

    def _scaled_hessian(self, theta, canonical):
        reference, weights, pair_weights = self._weights(theta, self._active_single, self._active_pair)
        singles = self._q[self._active_single]
        if canonical:
            singles = singles @ self._canonical_factor
        hessian = singles.T @ (weights[:, None] * singles)

        # Form q_a - q_b before contracting it. A Laplacian contraction can
        # subtract large, nearly equal terms along directions that are flat
        # for a given pair. Work in blocks to bound temporary array storage.
        nonzero = np.flatnonzero(pair_weights)
        for start in range(0, len(nonzero), PAIR_BLOCK_SIZE):
            block = nonzero[start:start + PAIR_BLOCK_SIZE]
            pairs = self._active_pair[block]
            differences = self._q[self._pair_a[pairs]] - self._q[self._pair_b[pairs]]
            if canonical:
                differences = differences @ self._canonical_factor
            hessian += differences.T @ (pair_weights[block, None] * differences)
        return reference, (hessian + hessian.T) * 0.5

    def hessian(self, theta=None):
        """Return the analytic Hessian with respect to the period-one angles."""
        reference, hessian = self._scaled_hessian(theta, canonical=False)
        log_scale = self._log_prefactor + 2 * np.log(FREQUENCY) + 4 * self._log_planck + reference
        with np.errstate(divide="ignore", under="ignore"):
            return np.sign(hessian) * np.exp(np.log(np.abs(hessian)) + log_scale)

    def masses(self, theta=None):
        """Return masses in eV with NaNs for negative squared masses."""
        reference, hessian = self._scaled_hessian(theta, canonical=True)
        squared_masses = eigvalsh(hessian)
        masses = np.full_like(squared_masses, np.nan)
        physical = squared_masses >= 0
        # This preserves M_pl * sqrt(eigenvalue of the dimensionless
        # canonically normalized Hessian) used by MassDist.py.
        log_scale = self._log_planck + 0.5 * (self._log_prefactor + 2 * np.log(FREQUENCY) + reference)
        with np.errstate(divide="ignore", under="ignore"):
            masses[physical] = np.exp(0.5 * np.log(squared_masses[physical]) + log_scale)
        return masses


def multiprecision_masses(q, tau, g_inv, volume, planck_mass_ev, *, W0, dps, divisor_volumes=None):
    """Assemble and diagonalize the origin Hessian in multiprecision."""
    from mpmath import mp

    ctx = mp.clone()
    ctx.dps = dps
    charge_rows = [
        tuple((index, ctx.mpf(value)) for index, value in enumerate(row) if value)
        for row in q.tolist()
    ]
    n_axions = q.shape[1]
    factor = ctx.cholesky(ctx.matrix(g_inv.tolist()))
    products = _charge_metric_products(ctx, charge_rows, factor)
    if divisor_volumes is None:
        moduli = [ctx.mpf(value) for value in tau.tolist()]
        actions = [ctx.fdot((value, moduli[i]) for i, value in row) for row in charge_rows]
    else:
        actions = [ctx.mpf(value) for value in divisor_volumes.tolist()]
    frequency = 2 * ctx.pi
    pi = ctx.pi
    suppressions = [ctx.exp(-frequency * action) for action in actions]
    w0 = ctx.mpf(str(W0))

    hessian = [[ctx.zero] * n_axions for _ in range(n_axions)]
    scale_bound = ctx.zero
    for a, qa in enumerate(charge_rows):
        single_weight = w0 * actions[a] * suppressions[a]
        _add_weighted_outer(hessian, qa, single_weight)
        scale_bound += abs(single_weight) * products[a][a]
        for b in range(a):
            qb = charge_rows[b]
            if qa == qb:
                continue
            product = products[a][b]
            bracket = pi * product + actions[a] + actions[b]
            if not bracket:
                continue
            weight = bracket * suppressions[a] * suppressions[b]
            _add_weighted_outer(hessian, _charge_difference(qa, qb), weight)
            norm_squared = products[a][a] + products[b][b] - 2 * product
            scale_bound += abs(weight) * max(norm_squared, ctx.zero)

    canonical = _canonical_hessian(ctx, factor, hessian)
    eigenvalues = ctx.eigsy(canonical, eigvals_only=True)

    # The sum of absolute weighted squared norms bounds the scale of the
    # terms before cancellation. Treat modes near its roundoff scale as unresolved.
    n_charges = len(charge_rows)
    term_count = n_charges * (n_charges + 1) // 2
    floor = 64 * (term_count + n_axions) * ctx.eps * scale_bound
    mass_scale = ctx.mpf(str(planck_mass_ev)) * frequency * ctx.sqrt(8 * ctx.pi) / ctx.mpf(float(volume))
    return np.array([
        float(mass_scale * ctx.sqrt(value)) if value > floor else np.nan
        for value in eigenvalues
    ], dtype=float)


def _charge_metric_products(ctx, charge_rows, factor):
    # Use sparse GLSM rows for Q L, then contract only one triangle
    # of (Q L) (Q L)^T. This preserves the original rounding order.
    factor_rows = factor.tolist()
    factor_columns = list(zip(*factor_rows))
    canonical = []
    for row in charge_rows:
        if len(row) == 1 and row[0][1] == 1:
            canonical.append(factor_rows[row[0][0]])
        else:
            canonical.append([ctx.fdot((value, column[i]) for i, value in row) for column in factor_columns])
    return [
        [ctx.fdot(row, other) for other in canonical[:i + 1]]
        for i, row in enumerate(canonical)
    ]


def _charge_difference(a, b):
    if len(a) == len(b) == 1 and a[0][0] != b[0][0]:
        other = (b[0][0], -b[0][1])
        return (a[0], other) if a[0][0] < other[0] else (other, a[0])
    values = dict(a)
    for i, value in b:
        values[i] = values.get(i, 0) - value
    return tuple((i, value) for i, value in sorted(values.items()) if value)


def _add_weighted_outer(hessian, charge, weight):
    """Accumulate one signed rank-one contribution in the lower triangle."""
    if not weight:
        return
    for row, (i, qi) in enumerate(charge):
        weighted = weight * qi
        for j, qj in charge[:row + 1]:
            hessian[i][j] += weighted * qj


def _canonical_hessian(ctx, factor, hessian):
    # Evaluate L^T H L, exploiting the zeros of the lower Cholesky
    # factor L and writing both halves of the symmetrized result together.
    n = factor.rows
    for i, row in enumerate(hessian):
        for j in range(i):
            hessian[j][i] = row[j]
    hessian_columns = list(zip(*hessian))
    factor_columns = [tuple(factor[i, j] for i in range(j, n)) for j in range(n)]
    transformed = [
        tuple(ctx.fdot(column, values[i:]) for values in hessian_columns)
        for i, column in enumerate(factor_columns)
    ]
    result = ctx.matrix(n)
    for i in range(n):
        for j in range(i + 1):
            value = ctx.fdot(transformed[i][j:], factor_columns[j])
            if i != j:
                value = (value + ctx.fdot(transformed[j][i:], factor_columns[i])) / 2
            result[i, j] = result[j, i] = value
    return result
