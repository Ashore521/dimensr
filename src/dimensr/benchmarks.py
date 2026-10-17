"""Small reproducible integration benchmarks; no claim of SOTA evaluation."""
from dataclasses import asdict, replace
import platform
import time
import numpy as np
from .units import ZERO
from .regressor import DimenSRRegressor

VARIANTS = {"full": {}, "no_rnn": {"use_rnn": False}, "no_gp": {"use_gp": False},
            "no_units": {"use_units": False}, "no_sd": {"use_size_diversity": False},
            "no_rad": {"use_rademacher": False}}
CASES = ("force", "kinetic", "nguyen1", "trigonometric")


def dataset(name, seed=0, n=160, noise=0.0):
    """Separate train/validation, IID test and OOD test draws. Noise on fitting data only."""
    if name not in CASES:
        raise ValueError(f"Unknown case: {name}")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 818]))
    if name == "force":
        names = ["mass", "acceleration"]
        units = [[0, 1, 0, 0, 0, 0, 0], [1, 0, -2, 0, 0, 0, 0]]
        target = [1, 1, -2, 0, 0, 0, 0]
        f = lambda x: x[:, 0] * x[:, 1]
        bounds, ood = (0.2, 2.0), (2.0, 4.0)
        truth = "mass * acceleration"
    elif name == "kinetic":
        names = ["mass", "velocity"]
        units = [[0, 1, 0, 0, 0, 0, 0], [1, 0, -1, 0, 0, 0, 0]]
        target = [2, 1, -2, 0, 0, 0, 0]
        f = lambda x: 0.5 * x[:, 0] * x[:, 1] ** 2
        bounds, ood = (0.2, 2.0), (2.0, 4.0)
        truth = "0.5 * mass * velocity**2"
    elif name == "nguyen1":
        names, units, target = ["x"], [ZERO], ZERO
        f = lambda x: x[:, 0] ** 3 + x[:, 0] ** 2 + x[:, 0]
        bounds, ood = (-1.0, 1.0), (1.0, 2.0)
        truth = "x**3 + x**2 + x"
    else:
        names, units, target = ["x"], [ZERO], ZERO
        f = lambda x: np.sin(x[:, 0]) + x[:, 0] ** 2
        bounds, ood = (-1.0, 1.0), (1.0, 2.0)
        truth = "sin(x) + x**2"
    X = rng.uniform(*bounds, (n, len(names)))
    y = f(X)
    y = y + noise * max(float(np.std(y)), 1e-12) * rng.normal(size=n)
    X_test = rng.uniform(*bounds, (n, len(names)))
    X_ood = rng.uniform(*ood, (n, len(names)))
    # Include exact zeros/boundaries that random floating-point draws almost surely miss.
    X_probe = (np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.], [2., 2.]])
               if len(names) == 2 else np.array([[-1.], [0.], [1.], [2.]]))
    return {"X": X, "y": y, "X_test": X_test, "y_test": f(X_test),
            "X_ood": X_ood, "y_ood": f(X_ood), "feature_names": names,
            "X_probe": X_probe, "y_probe": f(X_probe),
            "feature_units": units, "target_units": target, "truth": truth}


def run_benchmarks(config, cases=CASES, seeds=(0,), variants=("full",), noise=0.0, callback=None):
    import scipy
    import torch
    rows = []
    for case in cases:
        for seed in seeds:
            data = dataset(case, seed, noise=noise)
            for variant in variants:
                cfg = replace(config, random_state=seed, **VARIANTS[variant])
                start = time.monotonic()
                model = DimenSRRegressor(cfg, feature_units=data["feature_units"],
                                         target_units=data["target_units"], feature_names=data["feature_names"])
                # Evaluation failures are recorded rather than assigned fabricated scores.
                row = {"case": case, "seed": seed, "variant": variant, "noise": noise,
                       "truth": data["truth"], "config": asdict(cfg)}
                try:
                    model.fit(data["X"], data["y"])
                    row.update(status="ok", expression=str(model.expression_), nodes=model.best_.tree.size,
                               validation_rmse=model.validation_rmse_, diagnostics=model.diagnostics_,
                               model=model.best_.to_dict(data["feature_names"]))
                    for suffix in ("test", "ood", "probe"):
                        try:
                            error = model.predict(data[f"X_{suffix}"]) - data[f"y_{suffix}"]
                            rmse = float(np.sqrt(np.mean(error ** 2)))
                            normalized = rmse / float(np.std(data[f"y_{suffix}"]))
                            row[f"{suffix}_rmse"] = rmse
                            row[f"{suffix}_nrmse"] = normalized
                            row[f"{suffix}_r2"] = 1 - normalized ** 2
                        except ValueError as exc:
                            row[f"{suffix}_error"] = str(exc)
                    row["numerical_recovery"] = (row.get("test_nrmse", float("inf")) < 1e-6 and
                                                   row.get("ood_nrmse", float("inf")) < 1e-6 and
                                                   row.get("probe_nrmse", float("inf")) < 1e-6)
                except (ValueError, RuntimeError) as exc:
                    row.update(status="failed", error=str(exc))
                row["wall_seconds"] = time.monotonic() - start
                rows.append(row)
                if callback:
                    callback(rows)
    return {"scope": "small synthetic integration benchmark; numerical recovery is not symbolic equivalence",
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__},
            "results": rows}

