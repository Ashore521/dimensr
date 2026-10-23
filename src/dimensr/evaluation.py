from dataclasses import dataclass
from collections import Counter
import hashlib
import json
import numpy as np
from scipy.optimize import least_squares
from .complexity import local_rademacher, mixed_complexity
from .fitting import affine_constants, linear_solution, repair_domains
from .domains import domain_probes


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
    rademacher_complexity: float = 0.0

    def to_dict(self, feature_names=None):
        return {"tree": self.tree.to_dict(), "constants": self.constants.tolist(),
                "expression": str(self.tree.sympy(self.constants, feature_names)),
                "train_rmse": self.train_rmse, "train_nrmse": self.train_nrmse,
                "complexity": self.complexity, "rademacher": self.rademacher,
                "nodes": self.tree.size, "operator_types": self.tree.n_operator_types,
                "rademacher_complexity": self.rademacher_complexity}


class Evaluator:
    def __init__(self, X, y, config, *, input_bounds=None, X_domain=None):
        self.X, self.y, self.config = X, y, config
        self.scale = target_scale(y)
        self.cache = {}
        self.calls = 0
        self.constant_nfev = 0
        self.constant_residual_calls = 0
        self.proposals = 0
        self.cache_hits = 0
        self.linear_fits = 0
        self.domain_repairs = 0
        self.warm_starts = 0
        self.refinements = 0
        self.rademacher_evaluations = 0
        self.failures = Counter()
        self.optimizer_status = Counter()
        self.domain_rejections = 0
        rng = np.random.default_rng(np.random.SeedSequence([config.random_state, 721]))
        self.signs = rng.choice([-1.0, 1.0], (config.rademacher_draws, len(y)))
        self.X_probe, self.domain_audit = domain_probes(X, config, input_bounds=input_bounds, X_domain=X_domain)
        self.X_feasible = np.vstack((X, self.X_probe))

    def admissible(self, prediction):
        if not np.isfinite(prediction).all():
            return False
        if self.config.output_bounds is not None:
            lo, hi = self.config.output_bounds
            if lo is not None and np.any(prediction < lo):
                return False
            if hi is not None and np.any(prediction > hi):
                return False
        return True

    def __call__(self, tree, initial_constants=None, *, refine=False):
        self.proposals += 1
        tree, warm = tree.canonical(initial_constants)
        if tree in self.cache and not refine:
            self.cache_hits += 1
            return self.cache[tree]
        if tree not in self.cache and self.calls >= self.config.max_evaluations:
            return None
        previous = self.cache.get(tree)
        if tree not in self.cache:
            self.calls += 1
        else:
            self.refinements += 1
        cfg = self.config
        digest = hashlib.sha256(json.dumps(tree.to_dict(), sort_keys=True).encode()).digest()
        rng = np.random.default_rng(np.random.SeedSequence([cfg.random_state, int.from_bytes(digest[:8], "little")]))
        best = previous.constants.copy() if previous is not None else None
        best_error = previous.train_nrmse ** 2 if previous is not None else np.inf

        def evaluate(c):
            self.constant_residual_calls += 1
            return tree.evaluate(self.X, c)

        def error(c):
            values = evaluate(c)
            with np.errstate(all="ignore"):
                residual = (values - self.y) / self.scale
            return np.clip(np.nan_to_num(residual, nan=1e6, posinf=1e6, neginf=-1e6), -1e6, 1e6)

        def consider(c):
            nonlocal best, best_error
            pred = evaluate(c)
            if not self.admissible(pred):
                return
            if len(self.X_probe) and not self.admissible(tree.evaluate(self.X_probe, c)):
                self.domain_rejections += 1
                return
            with np.errstate(all="ignore"):
                mse = np.mean(((pred - self.y) / self.scale) ** 2)
            if np.isfinite(mse) and mse < best_error:
                best, best_error = c.copy(), float(mse)

        starts = [np.full(tree.n_constants, min(1.0, cfg.constant_bound * 0.5))]
        if initial_constants is not None:
            self.warm_starts += 1
            starts.insert(0, np.clip(warm, -cfg.constant_bound, cfg.constant_bound))
        if previous is not None:
            starts.insert(0, previous.constants.copy())
        linear = affine_constants(tree) and tree.n_constants > 0
        if linear:
            try:
                solution = linear_solution(tree, self.X, self.y, cfg.constant_bound, evaluate)
                if solution is not None:
                    self.linear_fits += 1
                    consider(solution)
            except (ValueError, np.linalg.LinAlgError):
                self.optimizer_status["linear_failure"] += 1
        if tree.n_constants:
            starts.extend(rng.uniform(-min(2.0, cfg.constant_bound * 0.9),
                                      min(2.0, cfg.constant_bound * 0.9), tree.n_constants)
                          for _ in range(cfg.constant_restarts - 1))
        feasible = []
        for initial in starts:
            repaired = repair_domains(tree, self.X_feasible, initial, cfg.constant_bound)
            if not np.array_equal(repaired, initial):
                self.domain_repairs += 1
            if np.isfinite(evaluate(repaired)).all():
                feasible.append(repaired)
        if not feasible and tree.n_constants:
            for _ in range(cfg.domain_restarts):
                initial = np.clip(rng.normal(size=tree.n_constants) * 10 ** rng.uniform(-3, 3),
                                  -cfg.constant_bound, cfg.constant_bound)
                repaired = repair_domains(tree, self.X_feasible, initial, cfg.constant_bound)
                if np.isfinite(evaluate(repaired)).all():
                    feasible.append(repaired)
        for initial in feasible:
            consider(initial)
            if tree.n_constants and not linear:
                try:
                    fit = least_squares(error, initial, bounds=(-cfg.constant_bound, cfg.constant_bound),
                                        max_nfev=cfg.polish_max_nfev if refine else cfg.constant_max_nfev,
                                        x_scale="jac")
                    self.constant_nfev += fit.nfev
                    self.optimizer_status[str(fit.status)] += 1
                    consider(fit.x)
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    self.optimizer_status["exception"] += 1
        if best is None:
            self.failures["no_feasible_start" if not feasible else "numeric_or_physical_rejection"] += 1
            self.cache[tree] = None
            return None
        def rad_evaluate(c):
            self.rademacher_evaluations += 1
            return tree.evaluate(self.X, c)

        rad = local_rademacher(tree, best, self.X, self.scale, self.signs,
                              radius=cfg.perturbation_radius, evaluate=rad_evaluate) if cfg.use_rademacher else 0.0
        rad_complexity = rad * np.sqrt(len(self.X)) / cfg.perturbation_radius if cfg.perturbation_radius else 0.0
        nrmse = float(np.sqrt(best_error))
        candidate = Candidate(tree, best, nrmse * self.scale, nrmse,
                              mixed_complexity(tree, rad_complexity, cfg.complexity_weights), rad, rad_complexity)
        self.cache[tree] = candidate
        return candidate

    def diagnostics(self):
        return {"proposals": self.proposals, "cache_hits": self.cache_hits,
                "linear_fits": self.linear_fits, "domain_repairs": self.domain_repairs,
                "warm_starts": self.warm_starts, "refinements": self.refinements,
                "failure_counts": dict(self.failures), "optimizer_status": dict(self.optimizer_status),
                "domain_probe_count": len(self.X_probe), "rademacher_evaluations": self.rademacher_evaluations,
                "domain_rejections": self.domain_rejections, "domain_audit": self.domain_audit,
                "rademacher_family": "local coefficient-Jacobian output ball",
                "rademacher_complexity_scale": float(np.sqrt(len(self.X)) / self.config.perturbation_radius)
                if self.config.perturbation_radius else 0.0}
