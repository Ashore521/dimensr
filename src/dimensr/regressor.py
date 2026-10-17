"""Public estimator and the two-stage neural/GP -> NSGA-II search."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
import warnings
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from .config import SearchConfig
from .units import ZERO, unit, encode
from .expression import Node
from .grammar import Grammar
from .policy import Policy
from .evaluation import Evaluator, Candidate
from .genetics import mutate, crossover
from .pareto import fronts, selection_order


def _unique(candidates):
    return list({c.tree: c for c in candidates if c is not None}.values())


def _ordered(candidates, cfg, multiobjective):
    candidates = _unique(candidates)
    if not multiobjective:
        return sorted(candidates, key=lambda c: (c.train_rmse, c.tree.size))
    indices = selection_order([(c.train_nrmse, c.complexity) for c in candidates],
                              [c.tree.size for c in candidates], cfg.use_size_diversity, cfg.size_radius)
    return [candidates[i] for i in indices]


class DimenSRRegressor(RegressorMixin, BaseEstimator):
    """Symbolic regressor with SI unit metadata and an independent validation split.

    X has shape (n_samples, n_features). Unit vectors have order L,M,T,I,Theta,N,J.
    A constant token denotes an independently fitted constant at each occurrence.
    Pass explicit X_val/y_val for grouped or time-dependent data.
    """
    def __init__(self, config=None, feature_units=None, target_units=None,
                 constant_units=None, feature_names=None, extra_units=None):
        self.config = config
        self.feature_units = feature_units
        self.target_units = target_units
        self.constant_units = constant_units
        self.feature_names = feature_names
        self.extra_units = extra_units

    def fit(self, X, y, *, X_val=None, y_val=None, callback=None):
        # Remove prior fitted state before a refit, including a failed refit.
        for name in list(vars(self)):
            if name.endswith("_"):
                delattr(self, name)
        cfg = replace(self.config) if self.config is not None else SearchConfig()
        cfg.validate()
        X, y = check_X_y(X, y, dtype=float, y_numeric=True)
        if len(X) < 5:
            raise ValueError("At least 5 samples are required.")
        n_features = X.shape[1]
        physical_features = tuple(unit(u) for u in self.feature_units) if self.feature_units is not None else (ZERO,) * n_features
        physical_target = unit(self.target_units) if self.target_units is not None else ZERO
        physical_constants = tuple(unit(u) for u in self.constant_units) if self.constant_units is not None else (ZERO,)
        extra = tuple(unit(u) for u in self.extra_units) if self.extra_units is not None else ()
        if len(physical_features) != n_features:
            raise ValueError("feature_units must match the number of columns.")
        names = list(self.feature_names) if self.feature_names is not None else [f"x{i}" for i in range(n_features)]
        if len(names) != n_features or len(set(names)) != n_features or any(not isinstance(n, str) or not n for n in names):
            raise ValueError("feature_names must contain distinct nonempty strings for all columns.")
        rng = np.random.default_rng(cfg.random_state)
        if (X_val is None) != (y_val is None):
            raise ValueError("Provide both X_val and y_val.")
        if X_val is None:
            split_rng = np.random.default_rng(np.random.SeedSequence([cfg.random_state, 911]))
            order = split_rng.permutation(len(X))
            n_val = max(1, int(np.ceil(len(X) * cfg.validation_fraction)))
            if len(X) - n_val < 2:
                raise ValueError("The validation fraction leaves fewer than 2 training samples.")
            self.validation_indices_, self.training_indices_ = order[:n_val], order[n_val:]
            X_val, y_val = X[self.validation_indices_], y[self.validation_indices_]
            X, y = X[self.training_indices_], y[self.training_indices_]
        else:
            X_val, y_val = check_X_y(X_val, y_val, dtype=float, y_numeric=True)
            if X_val.shape[1] != n_features:
                raise ValueError("Validation feature count does not match training data.")
            self.training_indices_ = np.arange(len(X))
            self.validation_indices_ = None
        units = physical_features if cfg.use_units else (ZERO,) * n_features
        target = physical_target if cfg.use_units else ZERO
        constants = physical_constants if cfg.use_units else (ZERO,) * len(physical_constants)
        grammar = Grammar(units, target, constants, cfg.operators, cfg.max_nodes, cfg.max_depth,
                          extra if cfg.use_units else ())
        evaluator = Evaluator(X, y, cfg)
        policy = Policy(len(grammar.tokens), cfg.hidden_size, cfg.learning_rate, cfg.random_state) if cfg.use_rnn else None
        self.history_ = []
        self.policy_ = policy
        start = time.monotonic()

        def record(phase, step, population, update=None):
            entry = {"phase": phase, "step": step, "evaluations": evaluator.calls,
                     "constant_nfev": evaluator.constant_nfev,
                     "constant_residual_calls": evaluator.constant_residual_calls,
                     "valid_candidates": sum(c is not None for c in evaluator.cache.values()),
                     "best_train_rmse": min(c.train_rmse for c in population),
                     "unique_sizes": len({c.tree.size for c in population}),
                     "elapsed_seconds": time.monotonic() - start}
            if update:
                entry.update(update)
            self.history_.append(entry)
            if callback:
                callback(entry.copy())

        def evolve(population, multiobjective):
            ranked = _ordered(population, cfg, multiobjective)
            children = []
            for _ in range(cfg.population_size):
                if evaluator.calls >= cfg.max_evaluations:
                    break
                # Tournament on rank/crowding order, or RMSE order in stage one.
                parent = ranked[int(rng.integers(len(ranked), size=3).min())]
                if rng.random() < cfg.crossover_rate:
                    donor = ranked[int(rng.integers(len(ranked), size=3).min())]
                    tree = crossover(parent.tree, donor.tree, grammar, rng)
                else:
                    tree = mutate(parent.tree, grammar, rng)
                candidate = evaluator(tree)
                if candidate is not None:
                    children.append(candidate)
            survivors = _ordered(population + children, cfg, multiobjective)[:cfg.population_size]
            return survivors, children

        # Stage one: repeated neural population seeding and open GP search.
        for epoch in range(cfg.epochs):
            sampled, traces, rewards = [], [], []
            for _ in range(cfg.population_size):
                if evaluator.calls >= cfg.max_evaluations:
                    break
                tree, trace = grammar.generate(rng, policy)
                candidate = evaluator(tree)
                traces.append(trace)
                rewards.append(0.0 if candidate is None else 1 / (1 + candidate.train_nrmse))
                if candidate is not None:
                    sampled.append(candidate)
            population = _ordered(sampled, cfg, False)
            if not population:
                population = _ordered([c for c in evaluator.cache.values() if c is not None], cfg, False)[:cfg.population_size]
            if not population:
                raise RuntimeError("No numerically valid candidates. Check units, domains, operators and search budget.")
            if cfg.use_gp:
                for _ in range(cfg.gp_generations):
                    population, _ = evolve(population, False)
            update = None
            if policy:
                elites = _ordered(population, cfg, False)[:max(1, int(cfg.population_size * cfg.risk_fraction))]
                update = policy.update(traces, rewards, [grammar.trace(c.tree) for c in elites],
                                       cfg.risk_fraction, cfg.entropy_weight, cfg.imitation_weight)
            record("neural_gp", epoch, population, update)
            if evaluator.calls >= cfg.max_evaluations:
                break

        archive = _ordered([c for c in evaluator.cache.values() if c is not None], cfg, False)
        requested = max(cfg.min_candidates, int(np.ceil(cfg.candidate_fraction * len(archive))))
        seeds = archive[:requested]
        if len(seeds) < cfg.min_candidates:
            warnings.warn(f"Only {len(seeds)} unique valid candidates found; requested minimum is {cfg.min_candidates}. "
                          "Increase the search budget for research experiments.", UserWarning)
        self.candidate_count_ = len(seeds)
        pool = list(seeds)
        population = _ordered(seeds, cfg, True)[:cfg.population_size]
        # Stage two: accuracy/complexity NSGA-II, with size-aware truncation.
        if cfg.use_gp:
            for generation in range(cfg.selection_generations):
                population, children = evolve(population, True)
                pool.extend(children)
                record("nsga2", generation, population)
                if evaluator.calls >= cfg.max_evaluations:
                    break
        pool = _unique(pool)
        front = fronts([(c.train_nrmse, c.complexity) for c in pool])[0]
        pareto = sorted([pool[i] for i in front], key=lambda c: (c.complexity, c.train_rmse))
        valid_front, scores, validation = [], [], []
        # Validation is first accessed for model scoring only after search has finished.
        for candidate in pareto:
            pred = candidate.tree.evaluate(X_val, candidate.constants)
            with np.errstate(all="ignore"):
                rmse = float(np.sqrt(np.mean((pred - y_val) ** 2)))
            if np.isfinite(rmse):
                valid_front.append(candidate)
                validation.append(rmse)
                scores.append(rmse / evaluator.scale + cfg.selection_complexity_weight * candidate.complexity / cfg.max_nodes)
        if not valid_front:
            raise RuntimeError("All Pareto models are invalid on validation data. Check domains or increase the search budget.")
        chosen = int(np.argmin(scores))
        self.best_ = valid_front[chosen]
        self.pareto_front_ = [dict(c.to_dict(names), validation_rmse=v, selection_score=s)
                              for c, v, s in zip(valid_front, validation, scores)]
        self.validation_rmse_ = validation[chosen]
        self.n_features_in_ = n_features
        self.feature_names_ = names
        self.feature_units_ = units
        self.target_units_ = target
        self.constant_units_ = constants
        self.config_ = cfg
        self.expression_ = self.best_.tree.sympy(self.best_.constants, names)
        self.diagnostics_ = {"evaluations": evaluator.calls, "constant_nfev": evaluator.constant_nfev,
                             "constant_residual_calls": evaluator.constant_residual_calls,
                             "valid_candidates": len([c for c in evaluator.cache.values() if c is not None]),
                             "screened_candidates": len(seeds), "candidate_minimum_met": len(seeds) >= cfg.min_candidates,
                             "pareto_count": len(pareto), "validation_valid_pareto_count": len(valid_front),
                             "training_rows": len(X), "validation_rows": len(X_val),
                             "budget_exhausted": evaluator.calls >= cfg.max_evaluations,
                             "elapsed_seconds": time.monotonic() - start}
        return self

    def predict(self, X):
        check_is_fitted(self, "best_")
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Feature count does not match the fitted model.")
        values = self.best_.tree.evaluate(X, self.best_.constants)
        if not np.isfinite(values).all():
            raise ValueError("Expression is undefined on these inputs (division, logarithm, square root or overflow).")
        return values

    def save(self, path):
        """Save an inference model as portable JSON, without executable pickle data."""
        check_is_fitted(self, "best_")
        data = {"format": "dimensr", "schema_version": 1, "package_version": "0.1.0",
                "config": asdict(self.config_), "n_features": self.n_features_in_,
                "feature_names": self.feature_names_, "feature_units": [encode(u) for u in self.feature_units_],
                "target_units": encode(self.target_units_), "constant_units": [encode(u) for u in self.constant_units_],
                "best": self.best_.to_dict(self.feature_names_), "validation_rmse": self.validation_rmse_,
                "pareto_front": self.pareto_front_, "history": self.history_, "diagnostics": self.diagnostics_}
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != "dimensr" or data.get("schema_version") != 1:
            raise ValueError("Unsupported model format or schema version.")
        cfg = SearchConfig(**data["config"])
        cfg.validate()
        model = cls(cfg, data["feature_units"], data["target_units"], data["constant_units"], data["feature_names"])
        model.config_ = cfg
        model.feature_units_ = tuple(unit(u) for u in data["feature_units"])
        model.target_units_ = unit(data["target_units"])
        model.constant_units_ = tuple(unit(u) for u in data["constant_units"])
        model.feature_names_ = data["feature_names"]
        model.n_features_in_ = data["n_features"]
        b = data["best"]
        tree = Node.from_dict(b["tree"])
        constants = np.asarray(b["constants"], dtype=float)
        if (tree.units != model.target_units_ or constants.ndim != 1 or len(constants) != tree.n_constants or
                not np.isfinite(constants).all() or len(model.feature_units_) != model.n_features_in_ or
                len(model.feature_names_) != model.n_features_in_):
            raise ValueError("Inconsistent model metadata.")
        for _, node in tree.walk():
            if not node.children:
                declared = model.feature_units_ if node.token.startswith("x") else model.constant_units_
                index = int(node.token[1:])
                if index >= len(declared) or node.units != declared[index]:
                    raise ValueError("Terminal units or feature index are inconsistent.")
        model.best_ = Candidate(tree, constants, b["train_rmse"], b["train_nrmse"], b["complexity"], b["rademacher"])
        model.expression_ = tree.sympy(constants, model.feature_names_)
        model.validation_rmse_ = data["validation_rmse"]
        model.pareto_front_, model.history_, model.diagnostics_ = data["pareto_front"], data["history"], data["diagnostics"]
        model.candidate_count_ = model.diagnostics_["screened_candidates"]
        return model
