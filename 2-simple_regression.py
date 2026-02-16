# /// script
# dependencies = ["pandas", "matplotlib", "scikit-learn", "numpy"]
# ///

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression

# Generate Regression Dataset with strong linear relationship
# n_samples=200, n_features=1, noise=15
X, y = make_regression(n_samples=200, n_features=1, noise=15, bias=100, random_state=42)

# Convert to DataFrame for assignment format
df = pd.DataFrame(X, columns=["Feature_X"])
df["Target_Y"] = y

X_train, X_test, y_train, y_test = train_test_split(
    df[["Feature_X"]], df["Target_Y"], test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Visualize
plt.scatter(X_test, y_test, color="black", alpha=0.7, label="Actual Data")
plt.plot(X_test, y_pred, color="blue", linewidth=3, label="Regression Line")
plt.title("Simple Regression Analysis (Synthetic)")
plt.xlabel("Feature X")
plt.ylabel("Target Y")
plt.legend()
plt.show()
