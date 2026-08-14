import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
import matplotlib.pyplot as plt


dataset = pd.read_csv(
    r"D:\Manas Doc\Pandas\Machine-Learning\Excel files\employee_test.csv"
)


X = dataset[
    [
        "Age",
        "Annual_Income",
        "Total_Purchases",
        "Avg_Order_Value",
        "Website_Visits"
    ]
]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


model = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)


dataset["Cluster"] = model.fit_predict(X_scaled)


print("\nCluster Sizes:")
print(dataset["Cluster"].value_counts().sort_index())


print("\nCluster Summary:")
summary = dataset.groupby("Cluster")[
    [
        "Age",
        "Annual_Income",
        "Total_Purchases",
        "Avg_Order_Value",
        "Website_Visits"
    ]
].mean()

print(summary.round(2))


Z = linkage(X_scaled, method="ward")

plt.figure(figsize=(12, 6))

dendrogram(
    Z,
    truncate_mode="lastp",
    p=30
)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Customer Groups")
plt.ylabel("Distance")
plt.grid(alpha=0.2)

plt.show()


plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=dataset,
    x="Annual_Income",
    y="Total_Purchases",
    hue="Cluster",
    palette="Set1",
    s=100
)

plt.title("Customer Segmentation using Hierarchical Clustering")
plt.xlabel("Annual Income")
plt.ylabel("Total Purchases")
plt.legend(title="Cluster")
plt.grid(alpha=0.2)

plt.show()