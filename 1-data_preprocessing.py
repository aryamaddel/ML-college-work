# /// script
# dependencies = ["pandas", "numpy"]
# ///

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load
df = pd.read_csv("Chocolate Sales.csv")

# Clean Amount: Remove '$' and ',' then convert to float
df["Amount"] = df["Amount"].replace({"\$": "", ",": ""}, regex=True).astype(float)

# Clean Date: Convert to datetime
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# Feature Engineering
df["Month"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year
df["Price_Per_Box"] = df["Amount"] / df["Boxes Shipped"]

# Drop original Date column and valid row check
# df = df.dropna() # Check if any NaNs exist

# Save
df.to_csv("chocolate_cleaned.csv", index=False)

print("Data Preprocessing Complete.")
print(df.head())
print(df.info())

# Visualize "Boxes Shipped" Distribution
plt.figure(figsize=(10, 6))
plt.hist(df["Boxes Shipped"], bins=20, color="skyblue", edgecolor="black")
plt.title("Distribution of Boxes Shipped")
plt.xlabel("Boxes Shipped")
plt.ylabel("Frequency")
plt.grid(axis="y", alpha=0.75)
plt.show()
