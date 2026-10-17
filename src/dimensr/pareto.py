"""NSGA-II with an optional, density-adjusted crowding truncation."""
import numpy as np


def fronts(objectives):
    values = np.asarray(objectives, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Objectives must be a finite 2D array.")
    dominates = (values[:, None] <= values[None, :]).all(2) & (values[:, None] < values[None, :]).any(2)
    counts = dominates.sum(0)
    current = np.flatnonzero(counts == 0).tolist()
    result = []
    while current:
        result.append(current)
        counts -= dominates[current].sum(0)
        counts[current] = -1
        current = np.flatnonzero(counts == 0).tolist()
    return result


def crowding(values):
    values = np.asarray(values, dtype=float)
    distance = np.zeros(len(values))
    if len(values) <= 2:
        distance[:] = np.inf
        return distance
    for j in range(values.shape[1]):
        order = np.argsort(values[:, j], kind="stable")
        span = values[order[-1], j] - values[order[0], j]
        if span == 0:
            continue
        distance[order[[0, -1]]] = np.inf
        distance[order[1:-1]] += (values[order[2:], j] - values[order[:-2], j]) / span
    return distance


def size_density(sizes, radius=1):
    sizes = np.asarray(sizes)
    return (np.abs(sizes[:, None] - sizes[None, :]) <= radius).sum(1) / (2 * radius + 1)


def selection_order(objectives, sizes, size_diversity=True, radius=1):
    values = np.asarray(objectives)
    density = size_density(sizes, radius)
    result = []
    for front in fronts(values):
        crowd = crowding(values[front])
        # Preserve infinite boundary crowding; penalize crowded size neighborhoods.
        if size_diversity:
            crowd = crowd / (1 + density[front] / max(float(density.mean()), 1e-12))
        local = sorted(range(len(front)), key=lambda i: (
            -crowd[i], density[front[i]] if size_diversity else 0, front[i]))
        result.extend(front[i] for i in local)
    return result
