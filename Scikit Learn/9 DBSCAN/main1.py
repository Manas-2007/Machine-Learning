import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import seaborn as sns
import matplotlib.pyplot as plt

# ========================= DATASET =========================

dataset = pd.read_csv(
    r"D:\Manas Doc\Pandas\Machine-Learning\Excel files\employee_test.csv"
)

print(dataset)

# ===================== FEATURE SELECTION ===================

X = dataset[
    [
        "Age",
        "Annual_Income",
        "Total_Purchases",
        "Avg_Order_Value",
        "Website_Visits"
    ]
]

# ======================= DATA SCALING =======================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ======================== DBSCAN MODEL ======================

model = DBSCAN(
    eps=0.7,
    min_samples=4
)

# ========================= TRAINING =========================

model.fit(X_scaled)

# ===================== CLUSTER ASSIGNMENT ==================

dataset["Cluster"] = model.labels_

# ======================= CLUSTER ANALYSIS ==================

print("\nCluster Sizes:")
print(dataset["Cluster"].value_counts().sort_index())

print("\nCluster Summary:")
print(
    dataset.groupby("Cluster")[
        [
            "Age",
            "Annual_Income",
            "Total_Purchases",
            "Avg_Order_Value",
            "Website_Visits"
        ]
    ].mean().round(2)
)

# ======================= NOISE ANALYSIS =====================

noise_count = (dataset["Cluster"] == -1).sum()

print("\nNoise / Outliers:", noise_count)

# ========================== GRAPH ===========================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=dataset,
    x="Annual_Income",
    y="Total_Purchases",
    hue="Cluster",
    palette="Set1",
    s=100
)

plt.title("DBSCAN Customer Clustering")
plt.xlabel("Annual Income")
plt.ylabel("Total Purchases")
plt.legend(title="Cluster (-1 = Noise)")
plt.grid(True, alpha=0.2)

plt.show()