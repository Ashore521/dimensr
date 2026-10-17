from dataclasses import dataclass
import math
from .expression import ARITY


@dataclass
class SearchConfig:
    population_size: int = 64
    epochs: int = 6
    gp_generations: int = 2
    selection_generations: int = 3
    max_nodes: int = 15
    max_depth: int = 6
    operators: tuple = ("add", "sub", "mul", "div", "square", "sin", "cos")
    hidden_size: int = 32
    learning_rate: float = 0.003
    risk_fraction: float = 0.3
    entropy_weight: float = 0.01
    imitation_weight: float = 0.1
    crossover_rate: float = 0.5
    candidate_fraction: float = 0.3
    min_candidates: int = 50
    complexity_weights: tuple = (1.0, 0.5, 2.0)
    selection_complexity_weight: float = 0.01
    size_radius: int = 1
    rademacher_draws: int = 16
    perturbations: int = 8
    perturbation_radius: float = 0.1
    constant_bound: float = 1e6
    constant_restarts: int = 2
    constant_max_nfev: int = 40
    validation_fraction: float = 0.2
    max_evaluations: int = 5000
    random_state: int = 0
    use_rnn: bool = True
    use_gp: bool = True
    use_units: bool = True
    use_size_diversity: bool = True
    use_rademacher: bool = True

    def validate(self):
        for name in ("use_rnn", "use_gp", "use_units", "use_size_diversity", "use_rademacher"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean.")
        for name in ("population_size", "epochs", "max_nodes", "max_depth", "hidden_size",
                     "min_candidates", "rademacher_draws", "perturbations", "constant_restarts",
                     "constant_max_nfev", "max_evaluations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.population_size < 4:
            raise ValueError("population_size must be at least 4.")
        for name in ("gp_generations", "selection_generations", "size_radius", "random_state"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
        for name in ("risk_fraction", "candidate_fraction", "validation_fraction"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} must be between 0 and 1.")
        if not 0 <= self.crossover_rate <= 1:
            raise ValueError("crossover_rate must be in [0, 1].")
        for name in ("entropy_weight", "imitation_weight", "selection_complexity_weight", "perturbation_radius"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        for name in ("learning_rate", "constant_bound"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if len(self.complexity_weights) != 3 or any(not math.isfinite(x) or x < 0 for x in self.complexity_weights):
            raise ValueError("complexity_weights must contain three finite nonnegative weights.")
        if not self.operators or len(set(self.operators)) != len(self.operators) or set(self.operators) - set(ARITY):
            raise ValueError(f"operators must be distinct supported names: {tuple(ARITY)}")

