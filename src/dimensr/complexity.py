"""Empirical Rademacher estimate for a local, linearized output-space ball."""
import numpy as np


def local_rademacher(tree, constants, X, y_scale, signs, directions=None, radius=0.1, *, evaluate=None):
    """R = radius / sqrt(n) * mean_sigma ||Q.T sigma||_2.

    Q spans the coefficient Jacobian on training inputs. The explicit function
    family is {Q a: ||a||_2 <= radius * sqrt(n)}, in normalized output units.
    Redundant/scaled coefficient coordinates do not change its span at regular
    points. This is NOT a bound for the nonlinear or structural search class.
    ``directions`` is accepted for compatibility with the 0.1 function signature.
    """
    if len(constants) == 0 or radius == 0:
        return 0.0
    evaluate = evaluate or (lambda c: tree.evaluate(X, c))
    base = evaluate(constants)
    if not np.isfinite(base).all():
        return float(radius)
    columns = []
    c = np.asarray(constants, dtype=float)
    for j in range(len(c)):
        step = np.cbrt(np.finfo(float).eps) * max(1.0, abs(c[j]))
        plus, minus = c.copy(), c.copy()
        plus[j] += step
        minus[j] -= step
        high, low = evaluate(plus), evaluate(minus)
        if np.isfinite(high).all() and np.isfinite(low).all():
            column = (high - low) / (2 * step * y_scale)
        elif np.isfinite(high).all():
            column = (high - base) / (step * y_scale)
        elif np.isfinite(low).all():
            column = (base - low) / (step * y_scale)
        else:
            return float(radius)
        norm = np.linalg.norm(column)
        if not np.isfinite(norm):
            return float(radius)
        if norm > 1e-12:
            columns.append(column / norm)
    if not columns:
        return 0.0
    q, singular, _ = np.linalg.svd(np.column_stack(columns), full_matrices=False)
    q = q[:, singular > singular[0] * 1e-6]
    return float(radius * np.linalg.norm(signs @ q, axis=1).mean() / np.sqrt(len(X)))


def mixed_complexity(tree, rademacher, weights):
    return float(weights[0] * tree.size + weights[1] * tree.n_operator_types + weights[2] * rademacher)
