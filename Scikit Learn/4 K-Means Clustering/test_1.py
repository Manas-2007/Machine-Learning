import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt


# =========================================================
# ========================= DATASET ========================
# =========================================================

data = {
    "Annual_Income": [
        22, 25, 28, 30, 32, 35, 38, 40, 42, 45,
        48, 50, 52, 55, 58, 60, 62, 65, 68, 70,
        72, 75, 78, 80, 82, 85, 88, 90, 92, 95,
        25, 28, 31, 34, 37, 41, 44, 47, 51, 54,
        57, 61, 66, 69, 73, 77, 81, 86, 89, 94
    ],

    "Spending_Score": [
        18, 22, 25, 28, 30, 33, 35, 38, 40, 42,
        45, 48, 50, 52, 55, 57, 60, 62, 65, 67,
        70, 72, 75, 78, 80, 82, 85, 87, 90, 93,
        75, 78, 72, 80, 76, 82, 74, 79, 77, 83,
        81, 85, 88, 86, 91, 89, 94, 92, 95, 97
    ]
}

dataset = pd.DataFrame(data)

print("\n========================= DATASET =========================")
print(dataset)

print("\nTotal Rows    :", len(dataset))
print("Total Features:", len(dataset.columns))


# =========================================================
# ===================== FEATURE SELECTION =================
# =========================================================

X = dataset[["Annual_Income", "Spending_Score"]]

print("\n==================== FEATURES SELECTED ===================")
print("Features used for clustering:")
print("1. Annual_Income")
print("2. Spending_Score")


# =========================================================
# ======================= DATA SCALING =====================
# =========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\n======================= DATA SCALING =====================")
print("StandardScaler applied successfully.")
print("Reason: K-Means is distance-based, so features should be")
print("on a comparable scale.")


# =========================================================
# ====================== K-MEANS MODEL =====================
# =========================================================

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

print("\n======================= MODEL SETUP ======================")
print("Algorithm      : K-Means Clustering")
print("Number of Clusters (K):", model.n_clusters)
print("n_init         :", model.n_init)
print("random_state   :", 42)


# =========================================================
# ========================= TRAINING =======================
# =========================================================

model.fit(X_scaled)

print("\n================ MODEL TRAINED SUCCESSFULLY ==============")


# =========================================================
# ===================== CLUSTER ASSIGNMENT =================
# =========================================================

dataset["Cluster"] = model.labels_

print("\n===================== CLUSTER ASSIGNMENT =================")
print(dataset)


# =========================================================
# ====================== CLUSTER SIZE =======================
# =========================================================

print("\n======================= CLUSTER SIZE =====================")

cluster_counts = dataset["Cluster"].value_counts().sort_index()

for cluster, count in cluster_counts.items():
    print(f"Cluster {cluster} : {count} customers")


# =========================================================
# ===================== CENTROIDS ===========================
# =========================================================

# Centroids are currently in scaled form.
# Convert them back to original units.

centroids = scaler.inverse_transform(model.cluster_centers_)

centroid_data = pd.DataFrame(
    centroids,
    columns=["Annual_Income", "Spending_Score"]
)

centroid_data.index.name = "Cluster"

print("\n======================= CENTROIDS =========================")
print(centroid_data)


# =========================================================
# ================= CLUSTER-WISE AVERAGES ==================
# =========================================================

print("\n=================== CLUSTER ANALYSIS =====================")

cluster_summary = dataset.groupby("Cluster")[
    ["Annual_Income", "Spending_Score"]
].mean()

print(cluster_summary.round(2))


# =========================================================
# ======================== INERTIA ==========================
# =========================================================

print("\n========================= INERTIA =========================")

print("Inertia :", round(model.inertia_, 2))

print("\nMeaning:")
print("Inertia measures how close the data points are to")
print("their assigned cluster centroids.")
print("Lower inertia generally means tighter clusters.")


# =========================================================
# ========================= GRAPH ===========================
# =========================================================

plt.figure(figsize=(10, 6))

dataset["Cluster_Label"] = dataset["Cluster"].map({
    0: "Low Income / Low Spending",
    1: "Medium Income / High Spending",
    2: "High Income / High Spending"
})

sns.scatterplot(
    data=dataset,
    x="Annual_Income",
    y="Spending_Score",
    hue="Cluster_Label",
    palette="Set1",
    s=100
)

# Plot centroids
plt.scatter(
    centroid_data["Annual_Income"],
    centroid_data["Spending_Score"],
    marker="X",
    s=300,
    color="black",
    edgecolor="white",
    linewidth=1.5,
    label="Centroids"
)

plt.title("K-Means Customer Segmentation")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

plt.legend()
plt.grid(alpha=0.2)

plt.show()