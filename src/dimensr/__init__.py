"""DimenSR: dimension-constrained neural-guided symbolic regression."""
from .config import SearchConfig
from .regressor import DimenSRRegressor

__all__ = ["SearchConfig", "DimenSRRegressor"]
__version__ = "0.3.0"
