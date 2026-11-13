"""Statistical kernels for the Figure 3 analysis.

Description
-----------
Statistical kernels for the Figure 3 analysis.

Methods
-------
1. Provide the reusable computations defined in this module.
2. Return results to the calling analysis code.

Inputs
------
- Data supplied through the module's public function arguments.

Outputs
-------
- In-memory values returned to the calling code.

Usage
-----
Import this module and call ``center_profiles(...)``; it has no command-line entry point.
"""
import numpy as np


def center_profiles(values):
    values = np.asarray(values, dtype=np.float64)
    centered = values - values.mean(axis=1, keepdims=True)
    if np.max(np.abs(centered.mean(axis=1)), initial=0) >= 1e-12:
        raise ValueError('Profile-wise mean centering failed')
    return centered


def weighted_pca(means, moments, counts=None):
    """Equal total subject weight, irrespective of valid vertex counts.

    means and moments are per-subject population first/second moments of
    profile-wise mean-centered MD. Normal PCA feature centering is applied
    here; no depth scaling or profile normalization is used.
    """
    if counts is None:
        counts = np.ones(len(means))
    weights = np.asarray(counts, dtype=float)
    weights /= weights.sum()
    mean = np.einsum('s,sd->d', weights, means)
    cov = np.einsum('s,sde->de', weights, moments) - np.outer(mean, mean)
    eigenvalues, vectors = np.linalg.eigh((cov + cov.T) / 2)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, vectors = np.maximum(eigenvalues[order], 0), vectors[:, order]
    contrast = np.zeros(21)
    contrast[16:] = 1 / 5
    contrast[:5] = -1 / 5
    sign_flipped = bool(vectors[:, 0] @ cov @ contrast < 0)
    if sign_flipped:
        vectors[:, 0] *= -1
    # Fix supplementary components deterministically at their largest loading.
    for k in range(1, 5):
        if vectors[np.argmax(np.abs(vectors[:, k])), k] < 0:
            vectors[:, k] *= -1
    w = vectors[:, 0]
    correlation = (w @ cov @ contrast) / np.sqrt((w @ cov @ w) * (contrast @ cov @ contrast))
    return mean, eigenvalues, vectors, sign_flipped, float(correlation)


def thickness_effect(tt, ty, yy, weights=None):
    """OLS after within-subject demeaning; all arguments are sufficient sums."""
    if weights is None:
        weights = np.ones(len(tt))
    weights = np.atleast_2d(weights)
    tt_sum, ty_sum, yy_sum = weights @ tt, weights @ ty, weights @ yy
    beta = ty_sum / tt_sum[:, None]
    standardized = ty_sum / np.sqrt(tt_sum[:, None] * yy_sum)
    partial_r2 = ty_sum ** 2 / (tt_sum[:, None] * yy_sum)
    return beta, standardized, partial_r2


def regional_partial_r2(regional_profiles, weights=None):
    """Subject + region FE partial R², without vertex pseudoreplication.

    A bootstrap multiplicity is a new copy of a subject, with its own intercept.
    Bilateral DK region identities are retained as 68 hemisphere-region cells.
    Incomplete grids dispatch to the exact observed-cell Gram-matrix fit.
    """
    y = np.asarray(regional_profiles, dtype=float)
    if not np.isfinite(y).all():
        stats = regional_sufficient_statistics(y)
        raw, _ = regional_effects_from_statistics(stats, weights)
        return raw
    y = y - y.mean(axis=1, keepdims=True)
    if weights is None:
        weights = np.ones((1, len(y)))
    weights = np.atleast_2d(weights)
    n = weights.sum(axis=1)
    sums = (weights @ y.reshape(len(y), -1)).reshape(len(weights), y.shape[1], y.shape[2])
    region_ss = (sums ** 2).sum(axis=1) / n[:, None]
    reduced_rss = weights @ (y ** 2).sum(axis=1)
    result = region_ss / reduced_rss
    if np.any(result < -1e-10) or np.any(result > 1 + 1e-10):
        raise ValueError('Invalid regional partial R²')
    return np.clip(result, 0, 1)


def regional_sufficient_statistics(profiles, thickness=None):
    """Within-subject region-dummy Gram matrices for an incomplete region grid.

    Missing cells contribute no observations. No regional values are imputed.
    The observed-cell pattern must be shared by all depths (whole profiles).
    """
    y = np.asarray(profiles, dtype=float)
    valid = np.isfinite(y).all(axis=2)
    if np.any(np.isfinite(y).any(axis=2) != valid):
        raise ValueError('Regional missingness must be whole-profile')
    count = valid.sum(axis=1)
    if np.any(count < 2):
        raise ValueError('Each subject needs at least two observed regions')
    obs = valid.astype(float)
    mean = np.nansum(y, axis=1) / count[:, None]
    yc = np.where(valid[:, :, None], y - mean[:, None, :], 0.)
    gram = -obs[:, :, None] * obs[:, None, :] / count[:, None, None]
    diagonal = np.arange(y.shape[1])
    gram[:, diagonal, diagonal] += obs
    result = dict(gram=gram, y=yc, yy=(yc*yc).sum(axis=1))
    if thickness is not None:
        t = np.asarray(thickness, dtype=float)
        if not np.array_equal(np.isfinite(t), valid):
            raise ValueError('Thickness and MD regional support differ')
        tm = np.nansum(t, axis=1) / count
        tc = np.where(valid, t-tm[:, None], 0.)
        result.update(t=tc, tt=(tc*tc).sum(axis=1), ty=(tc[:, :, None]*yc).sum(axis=1))
    return result


def regional_effects_from_statistics(stats, weights=None, beta=None):
    """Exact subject+region FE partial R², raw and optionally adjusted.

    Subject intercepts are eliminated analytically using only observed cells.
    Beta can vary by bootstrap replicate. One joint solve covers every depth.
    """
    y = stats['y']; n, r, d = y.shape
    w = np.ones((1,n)) if weights is None else np.atleast_2d(weights).astype(float)
    gram = (w @ stats['gram'].reshape(n,-1)).reshape(len(w),r,r)
    rhs = (w @ y.reshape(n,-1)).reshape(len(w),r,d)
    reduced = w @ stats['yy']
    if beta is not None:
        beta = np.broadcast_to(np.asarray(beta), (len(w),d))
        t_rhs = w @ stats['t']
        adjusted_rhs = rhs - t_rhs[:, :, None]*beta[:,None,:]
        adjusted_reduced = reduced - 2*beta*(w@stats['ty']) + beta*beta*(w@stats['tt'])[:,None]
        joint = np.concatenate([rhs,adjusted_rhs],axis=2)
    else:
        joint = rhs
    # Add a constraint for the unidentifiable common region intercept.
    constrained = gram + np.ones((r,r))/r
    try:
        coefficients = np.linalg.solve(constrained,joint)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(gram,hermitian=True) @ joint
    explained = (joint*coefficients).sum(axis=1)
    raw = explained[:,:d]/reduced
    adjusted = None if beta is None else explained[:,d:]/adjusted_reduced
    for result in (raw,adjusted):
        if result is not None and (not np.isfinite(result).all() or np.any(result < -1e-8) or np.any(result > 1+1e-8)):
            raise ValueError('Invalid incomplete-grid regional partial R²')
    return np.clip(raw,0,1), None if adjusted is None else np.clip(adjusted,0,1)
