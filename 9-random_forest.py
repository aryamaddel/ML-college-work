# /// script
# dependencies = ["pandas", "matplotlib", "scikit-learn", "numpy"]
# ///

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt

# Load Wine dataset (Good for Random Forest Classification)
data = load_wine()
X = data.data
y = data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy on Wine Dataset: {acc:.4f}")

# Feature Importance
importances = rf.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.title("Random Forest Feature Importance (Wine)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

print(
    "\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0)
)
