from dataclasses import dataclass
import hashlib
import json
import numpy as np
from scipy.optimize import least_squares
from .complexity import local_rademacher, mixed_complexity


def target_scale(y):
    std = float(np.std(y))
    return std if std > 1e-12 else max(float(np.sqrt(np.mean(y ** 2))), 1.0)


@dataclass
class Candidate:
    tree: object
    constants: np.ndarray
    train_rmse: float
    train_nrmse: float
    complexity: float
    rademacher: float

    def to_dict(self, feature_names=None):
        return {"tree": self.tree.to_dict(), "constants": self.constants.tolist(),
                "expression": str(self.tree.sympy(self.constants, feature_names)),
                "train_rmse": self.train_rmse, "train_nrmse": self.train_nrmse,
                "complexity": self.complexity, "rademacher": self.rademacher,
                "nodes": self.tree.size, "operator_types": self.tree.n_operator_types}


class Evaluator:
    def __init__(self, X, y, config):
        self.X, self.y, self.config = X, y, config
        self.scale = target_scale(y)
        self.cache = {}
        self.calls = 0
        self.constant_nfev = 0
        self.constant_residual_calls = 0
        rng = np.random.default_rng(np.random.SeedSequence([config.random_state, 721]))
        self.signs = rng.choice([-1.0, 1.0], (config.rademacher_draws, len(y)))
        self.directions = rng.choice([-1.0, 1.0], (config.perturbations, config.max_nodes))

    def __call__(self, tree):
        if tree in self.cache:
            return self.cache[tree]
        if self.calls >= self.config.max_evaluations:
            return None
        self.calls += 1
        cfg = self.config
        digest = hashlib.sha256(json.dumps(tree.to_dict(), sort_keys=True).encode()).digest()
        rng = np.random.default_rng(np.random.SeedSequence([cfg.random_state, int.from_bytes(digest[:8], "little")]))
        best, best_error = None, np.inf

        def error(c):
            self.constant_residual_calls += 1
            values = tree.evaluate(self.X, c)
            with np.errstate(all="ignore"):
                residual = (values - self.y) / self.scale
            return np.clip(np.nan_to_num(residual, nan=1e6, posinf=1e6, neginf=-1e6), -1e6, 1e6)

        starts = [np.full(tree.n_constants, min(1.0, cfg.constant_bound * 0.5))]
        if tree.n_constants:
            starts.extend(rng.uniform(-min(2.0, cfg.constant_bound * 0.9),
                                      min(2.0, cfg.constant_bound * 0.9), tree.n_constants)
                          for _ in range(cfg.constant_restarts - 1))
        for initial in starts:
            options = [initial]
            if tree.n_constants:
                try:
                    fit = least_squares(error, initial, bounds=(-cfg.constant_bound, cfg.constant_bound),
                                        max_nfev=cfg.constant_max_nfev)
                    self.constant_nfev += fit.nfev
                    options.append(fit.x)
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    pass
            for c in options:
                pred = tree.evaluate(self.X, c)
                with np.errstate(all="ignore"):
                    mse = np.mean(((pred - self.y) / self.scale) ** 2)
                if np.isfinite(mse) and mse < best_error:
                    best, best_error = c.copy(), float(mse)
        if best is None:
            self.cache[tree] = None
            return None
        rad = local_rademacher(tree, best, self.X, self.scale, self.signs,
                              self.directions, cfg.perturbation_radius) if cfg.use_rademacher else 0.0
        nrmse = float(np.sqrt(best_error))
        candidate = Candidate(tree, best, nrmse * self.scale, nrmse,
                              mixed_complexity(tree, rad, cfg.complexity_weights), rad)
        self.cache[tree] = candidate
        return candidate
