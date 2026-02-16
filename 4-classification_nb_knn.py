# /// script
# dependencies = ["pandas", "matplotlib", "scikit-learn", "numpy"]
# ///

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_digits

# Load Digits Dataset (Good for KNN/Naive Bayes)
# Contains 8x8 images of digits 0-9
data = load_digits()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

print("Classification on Digits Dataset (Mnist-like):")
print(f"Naive Bayes Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}")
print(f"KNN Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")

print("\nKNN Report:\n", classification_report(y_test, y_pred_knn))
