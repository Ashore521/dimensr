"""Unlabelled domain probes derived from training inputs or declared input ranges."""
import hashlib
import itertools
import numpy as np


def domain_probes(X, config, *, input_bounds=None, X_domain=None):
    """Check permitted boundary points, never consult validation or target values.

    A declared box must be a valid domain for the intended relationship. For
    coupled constraints (e.g. m1+m2>0), supply explicit permitted X_domain points.
    Disable boundary_probes and set domain_probe_count=0 when a bounding box or
    interpolation would generate inputs outside the permitted physical domain.
    Finite probes are a screening mechanism, not a proof over a continuous domain.
    """
    n = X.shape[1]
    bounds = np.column_stack((X.min(axis=0), X.max(axis=0)))
    if input_bounds is not None:
        bounds = np.asarray(input_bounds, dtype=float)
        if bounds.shape != (n, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] > bounds[:, 1]):
            raise ValueError("input_bounds must contain one finite (lower, upper) pair per feature.")
        if np.any(X < bounds[:, 0]) or np.any(X > bounds[:, 1]):
            raise ValueError("Declared input_bounds must contain all training inputs.")
    center = (bounds[:, 0] + bounds[:, 1]) / 2
    axes = [sorted(set([lo, hi, mid] + ([0.] if lo <= 0 <= hi else [])))
            for (lo, hi), mid in zip(bounds, center)]
    boundary = []
    # Coordinate faces are affordable even with many variables.
    for i, values in enumerate(axes if config.boundary_probes or input_bounds is not None else []):
        for value in values:
            row = center.copy()
            row[i] = value
            boundary.append(row)
    if (config.boundary_probes or input_bounds is not None) and np.prod([len(a) for a in axes], dtype=float) <= 256:
        boundary.extend(itertools.product(*axes))
    elif config.boundary_probes or input_bounds is not None:
        rng = np.random.default_rng(np.random.SeedSequence([config.random_state, 724]))
        boundary.extend(np.column_stack([rng.choice(a, 256) for a in axes]))
    rng = np.random.default_rng(np.random.SeedSequence([config.random_state, 723]))
    pairs = rng.integers(len(X), size=(config.domain_probe_count, 2))
    weight = rng.uniform(0, 1, (config.domain_probe_count, 1))
    interior = weight * X[pairs[:, 0]] + (1 - weight) * X[pairs[:, 1]]
    explicit = np.empty((0, n)) if X_domain is None else np.asarray(X_domain, dtype=float)
    if explicit.ndim != 2 or explicit.shape[1] != n or not np.isfinite(explicit).all():
        raise ValueError("X_domain must be a finite two-dimensional array with the training feature count.")
    probes = np.unique(np.vstack((np.asarray(boundary).reshape(-1, n), interior, explicit)), axis=0)
    source = "declared_box" if input_bounds is not None else "training_box" if config.boundary_probes else "explicit_and_interpolation"
    return probes, dict(source=source, automatic_boundaries=config.boundary_probes,
                        input_bounds=bounds.tolist(), explicit_rows=len(explicit), total_rows=len(probes),
                        points_sha256=hashlib.sha256(probes.astype("<f8").tobytes()).hexdigest(),
                        guarantee="finite probes only; no continuous-domain proof")
