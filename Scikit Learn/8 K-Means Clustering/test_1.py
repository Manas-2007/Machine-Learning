import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

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

X = dataset[["Annual_Income", "Spending_Score"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

model.fit(X_scaled)

dataset["Cluster"] = model.labels_

cluster_counts = dataset["Cluster"].value_counts().sort_index()

centroids = scaler.inverse_transform(model.cluster_centers_)

centroid_data = pd.DataFrame(
    centroids,
    columns=["Annual_Income", "Spending_Score"]
)

centroid_data.index.name = "Cluster"

cluster_summary = dataset.groupby("Cluster")[
    ["Annual_Income", "Spending_Score"]
].mean().round(2)

print(dataset)
print("\nCluster Sizes:")
print(cluster_counts)

print("\nCluster Summary:")
print(cluster_summary)

print("\nCentroids:")
print(centroid_data)

print("\nInertia:", round(model.inertia_, 2))

dataset["Cluster_Label"] = dataset["Cluster"].map({
    0: "Low Income / Low Spending",
    1: "Medium Income / High Spending",
    2: "High Income / High Spending"
})

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=dataset,
    x="Annual_Income",
    y="Spending_Score",
    hue="Cluster_Label",
    palette="Set1",
    s=100
)

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