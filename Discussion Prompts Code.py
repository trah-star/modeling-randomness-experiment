import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

_, _, true_coef = make_regression(
    n_samples=200,
    n_features=5,
    n_informative=3,
    noise=20,
    coef=True,
    random_state=42
)

coefs, errors = [], []

for seed in range(100):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(200, 5))
    y = X @ true_coef + rng.normal(scale=20, size=200)

    model = LinearRegression().fit(X, y)
    coefs.append(model.coef_)
    errors.append(mean_squared_error(y, model.predict(X)))

coefs = np.array(coefs)
errors = np.array(errors)

print("true:", np.round(true_coef, 2))
print("mean:", np.round(coefs.mean(axis=0), 2))
print("std:", np.round(coefs.std(axis=0), 2))
print("MSE mean:", round(errors.mean(), 1))
print("MSE std:", round(errors.std(), 1))

fig, ax = plt.subplots(figsize=(8, 5))
ax.boxplot(coefs, labels=["X1", "X2", "X3", "X4", "X5"])
ax.scatter(range(1, 6), true_coef, label="True value", zorder=3)
ax.axhline(0, linewidth=0.8)
ax.set_title("Estimated Coefficients Across 100 Datasets")
ax.set_ylabel("Coefficient")
ax.legend()
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(errors, bins=15)
ax.axvline(errors.mean(), linestyle="--", linewidth=1.5,
           label=f"Mean MSE = {errors.mean():.1f}")
ax.set_title("Model Error (MSE) Across 100 Datasets")
ax.set_xlabel("Mean Squared Error")
ax.set_ylabel("Number of Datasets")
ax.legend()
plt.tight_layout()
plt.show()
