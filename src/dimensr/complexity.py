"""Rademacher estimate for an explicitly finite, symmetric local function family."""
import numpy as np


def local_rademacher(tree, constants, X, y_scale, signs, directions, radius=0.1):
    if len(constants) == 0 or radius == 0:
        return 0.0
    base = tree.evaluate(X, constants)
    if not np.isfinite(base).all():
        return 1.0
    correlations = [np.zeros(len(signs))]  # Include the zero function.
    c = np.asarray(constants)
    for direction in directions[:, :len(c)]:
        shifted = tree.evaluate(X, c + radius * np.maximum(1, np.abs(c)) * direction)
        if not np.isfinite(shifted).all():
            # Invalid local perturbations are conservatively penalized, not silently discarded.
            return 1.0
        g = np.clip((shifted - base) / y_scale, -1.0, 1.0)
        correlations.append(np.abs(signs @ g) / len(X))
    return float(np.max(correlations, axis=0).mean())


def mixed_complexity(tree, rademacher, weights):
    return float(weights[0] * tree.size + weights[1] * tree.n_operator_types + weights[2] * rademacher)
