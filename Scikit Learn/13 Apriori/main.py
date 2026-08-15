import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


# =========================================================
# TRANSACTION DATA
# =========================================================

transactions = [
    ["Milk", "Bread", "Butter"],
    ["Bread", "Milk"],
    ["Milk", "Eggs"],
    ["Bread", "Butter"],
    ["Milk", "Bread", "Eggs"],
    ["Bread", "Butter", "Jam"],
    ["Milk", "Bread", "Butter", "Eggs"],
    ["Eggs", "Bread"],
    ["Milk", "Cereal"],
    ["Bread", "Milk", "Butter"],

    ["Milk", "Bread", "Eggs"],
    ["Bread", "Butter"],
    ["Milk", "Bread"],
    ["Milk", "Eggs", "Cereal"],
    ["Bread", "Jam"],
    ["Milk", "Bread", "Butter"],
    ["Eggs", "Bread", "Butter"],
    ["Milk", "Cereal"],
    ["Bread", "Butter", "Jam"],
    ["Milk", "Bread", "Eggs"],

    ["Milk", "Bread", "Butter"],
    ["Bread", "Eggs"],
    ["Milk", "Eggs"],
    ["Bread", "Butter", "Jam"],
    ["Milk", "Bread"],
    ["Milk", "Bread", "Butter", "Jam"],
    ["Eggs", "Bread", "Butter"],
    ["Milk", "Cereal", "Bread"],
    ["Bread", "Butter"],
    ["Milk", "Eggs", "Bread"],

    ["Milk", "Bread", "Butter"],
    ["Bread", "Jam"],
    ["Milk", "Bread", "Eggs"],
    ["Eggs", "Butter"],
    ["Milk", "Cereal"],
    ["Bread", "Butter", "Jam"],
    ["Milk", "Bread"],
    ["Milk", "Eggs", "Bread"],
    ["Bread", "Butter"],
    ["Milk", "Bread", "Butter", "Eggs"],

    ["Milk", "Bread"],
    ["Bread", "Butter", "Jam"],
    ["Milk", "Eggs"],
    ["Bread", "Eggs", "Butter"],
    ["Milk", "Bread", "Butter"],
    ["Cereal", "Milk"],
    ["Bread", "Jam"],
    ["Milk", "Bread", "Eggs"],
    ["Bread", "Butter"],
    ["Milk", "Eggs", "Cereal"],

    ["Milk", "Bread", "Butter"],
    ["Bread", "Eggs"],
    ["Milk", "Bread"],
    ["Eggs", "Butter", "Jam"],
    ["Milk", "Cereal", "Bread"],
    ["Bread", "Butter"],
    ["Milk", "Bread", "Eggs"],
    ["Bread", "Jam"],
    ["Milk", "Eggs"],
    ["Milk", "Bread", "Butter"],

    ["Bread", "Butter", "Jam"],
    ["Milk", "Bread"],
    ["Eggs", "Bread", "Butter"],
    ["Milk", "Cereal"],
    ["Bread", "Eggs"],
    ["Milk", "Bread", "Butter"],
    ["Milk", "Eggs", "Bread"],
    ["Bread", "Butter"],
    ["Cereal", "Milk", "Eggs"],
    ["Bread", "Jam"],

    ["Milk", "Bread", "Butter"],
    ["Bread", "Eggs"],
    ["Milk", "Bread"],
    ["Milk", "Eggs", "Cereal"],
    ["Bread", "Butter", "Jam"],
    ["Milk", "Bread", "Eggs"],
    ["Eggs", "Bread", "Butter"],
    ["Milk", "Cereal"],
    ["Bread", "Butter"],
    ["Milk", "Bread", "Butter", "Jam"],

    ["Milk", "Eggs"],
    ["Bread", "Butter"],
    ["Milk", "Bread", "Eggs"],
    ["Bread", "Jam"],
    ["Milk", "Bread", "Butter"],
    ["Eggs", "Bread"],
    ["Milk", "Cereal", "Bread"],
    ["Bread", "Butter", "Jam"],
    ["Milk", "Eggs", "Bread"],
    ["Bread", "Butter"],

    ["Milk", "Bread", "Butter"],
    ["Eggs", "Bread"],
    ["Milk", "Bread"],
    ["Bread", "Butter", "Jam"],
    ["Milk", "Eggs"],
    ["Cereal", "Milk", "Bread"],
    ["Bread", "Eggs", "Butter"],
    ["Milk", "Bread", "Butter"],
    ["Bread", "Jam"],
    ["Milk", "Eggs", "Cereal"]
]


# =========================================================
# CONVERT TRANSACTIONS INTO ONE-HOT DATA
# =========================================================

encoder = TransactionEncoder()

encoded_data = encoder.fit(transactions).transform(transactions)

transaction_data = pd.DataFrame(
    encoded_data,
    columns=encoder.columns_
)


# =========================================================
# APRIORI
# =========================================================

frequent_itemsets = apriori(
    transaction_data,
    min_support=0.20,
    use_colnames=True
)


# =========================================================
# ASSOCIATION RULES
# =========================================================

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.50
)


# =========================================================
# DISPLAY RESULTS
# =========================================================

print("\nFREQUENT ITEMSETS")
print(frequent_itemsets.sort_values(
    by="support",
    ascending=False
))


print("\nASSOCIATION RULES")

result = rules[
    [
        "antecedents",
        "consequents",
        "support",
        "confidence",
        "lift"
    ]
].sort_values(
    by="lift",
    ascending=False
)

print(result)