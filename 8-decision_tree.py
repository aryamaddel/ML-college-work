# /// script
# dependencies = ["pandas", "matplotlib", "scikit-learn", "numpy"]
# ///

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load Iris dataset (Perfect for Decision Trees)
data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Decision Tree
clf = DecisionTreeClassifier(random_state=42, max_depth=3)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)
print(f"Decision Tree Accuracy on Iris Dataset: {accuracy_score(y_test, y_pred):.2f}")

# Visualize Tree
plt.figure(figsize=(10, 8))
plot_tree(
    clf,
    feature_names=data.feature_names,
    filled=True,
    class_names=data.target_names,
    rounded=True,
)
plt.title("Decision Tree Visualization (Iris Data)")
plt.show()
