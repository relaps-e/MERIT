"""Observed-record regression with row-sparse feature weights."""

import numpy as np


def fit_feature_weights(source, performance, penalty=0.001, max_iter=2000, tol=1e-6):
    """Fit W using min-max training performance and an L2,1 penalty.

    Minimize mean observed squared error plus penalty * sum_k ||W[k]||_2.
    Missing performance entries are excluded from both loss and normalization.
    No intercept or target records are used.
    """
    source = np.asarray(source, dtype=np.float64)
    performance = np.asarray(performance, dtype=np.float64)
    if source.ndim != 2 or performance.ndim != 2 or len(source) != len(performance):
        raise ValueError("Expected feature and performance matrices with matching source rows")
    if penalty < 0 or max_iter < 1 or tol <= 0:
        raise ValueError("penalty must be nonnegative; max_iter and tol must be positive")

    observed = np.isfinite(performance)
    rows = observed.any(axis=1)
    if not rows.any():
        raise ValueError("Feature selection requires observed training performance")
    source, performance, observed = source[rows], performance[rows], observed[rows]
    if not np.isfinite(source).all():
        raise ValueError("Source features must be finite")
    lower = np.where(observed, performance, np.inf).min(axis=1, keepdims=True)
    upper = np.where(observed, performance, -np.inf).max(axis=1, keepdims=True)
    span = upper - lower
    normalized = np.divide(
        np.where(observed, performance, lower) - lower,
        span,
        out=np.zeros_like(performance),
        where=span > 0,
    )

    # The largest eigenvalue of X X^T gives a Lipschitz bound for the masked loss.
    spectral = np.linalg.eigvalsh(source @ source.T)[-1]
    weights = np.zeros((source.shape[1], performance.shape[1]), dtype=np.float64)
    if spectral == 0:
        return weights
    observed_count = int(observed.sum())
    step = observed_count / (2.0 * spectral)
    momentum, acceleration = weights.copy(), 1.0
    for _ in range(max_iter):
        residual = np.where(observed, source @ momentum - normalized, 0.0)
        proposal = momentum - (2.0 * step / observed_count) * (source.T @ residual)
        norms = np.linalg.norm(proposal, axis=1, keepdims=True)
        shrinkage = np.divide(
            step * penalty, norms, out=np.ones_like(norms), where=norms > 0,
        )
        updated = proposal * np.maximum(0.0, 1.0 - shrinkage)
        change = np.linalg.norm(updated - weights)
        if change <= tol * max(1.0, np.linalg.norm(weights)):
            weights = updated
            break
        next_acceleration = (1.0 + np.sqrt(1.0 + 4.0 * acceleration**2)) / 2.0
        momentum = updated + ((acceleration - 1.0) / next_acceleration) * (updated - weights)
        weights, acceleration = updated, next_acceleration
    return weights


def select_features(source, performance, count=32, penalty=0.001, max_iter=2000, tol=1e-6):
    """Return feature indices with the largest fitted row norms, in column order."""
    source = np.asarray(source)
    if not 1 <= count <= source.shape[1]:
        raise ValueError("count must be between one and the number of source features")
    if count == source.shape[1]:
        return np.arange(count, dtype=np.int32)
    weights = fit_feature_weights(source, performance, penalty, max_iter, tol)
    importance = np.linalg.norm(weights, axis=1)
    if not importance.any():
        raise ValueError("All feature weights are zero; reduce penalty or check source features")
    selected = np.argsort(-importance, kind="mergesort")[:count]
    return np.sort(selected).astype(np.int32)
