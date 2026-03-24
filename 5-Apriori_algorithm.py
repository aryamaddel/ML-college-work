# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas",
#     "mlxtend"
# ]
# ///

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# -----------------------------
# STEP 1: Load Dataset
# -----------------------------
file_path = "Groceries_dataset.csv"  # update path if needed
df = pd.read_csv(file_path)

print("Original Data:")
print(df.head())

# -----------------------------
# STEP 2: Create Transactions
# -----------------------------
# Combine Member_number + Date into a transaction
transactions_df = df.groupby(['Member_number', 'Date'])['itemDescription'].apply(list)

transactions = transactions_df.tolist()

print("\nSample Transactions:")
print(transactions[:5])

# -----------------------------
# STEP 3: One-Hot Encoding
# -----------------------------
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)

basket = pd.DataFrame(te_array, columns=te.columns_)

print("\nOne-hot Encoded Basket:")
print(basket.head())

# -----------------------------
# STEP 4: Apply Apriori
# -----------------------------
frequent_itemsets = apriori(
    basket,
    min_support=0.01,
    use_colnames=True
)

print("\nFrequent Itemsets:")
print(frequent_itemsets.head())

# -----------------------------
# STEP 5: Generate Association Rules
# -----------------------------
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.3
)

# Keep only meaningful rules
rules = rules[rules['lift'] > 1]

# -----------------------------
# STEP 6: Results
# -----------------------------
print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# -----------------------------
# STEP 7: Sort by Lift
# -----------------------------
rules_sorted = rules.sort_values(by='lift', ascending=False)

print("\nTop Rules by Lift:")
print(rules_sorted[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())