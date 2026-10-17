"""Run after `pip install -e .`. Fits F = m*a from synthetic observations."""
import numpy as np
from dimensr import DimenSRRegressor, SearchConfig

rng = np.random.default_rng(42)
X = rng.uniform(0.2, 2.0, (160, 2))
y = X[:, 0] * X[:, 1]
model = DimenSRRegressor(
    SearchConfig(random_state=42),
    feature_names=["mass", "acceleration"],
    feature_units=[[0, 1, 0, 0, 0, 0, 0], [1, 0, -2, 0, 0, 0, 0]],
    target_units=[1, 1, -2, 0, 0, 0, 0],
).fit(X, y)
print(model.expression_)
print(model.predict([[3.0, 2.0]]))
model.save("force_model.json")

