import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

model.fit(X_scaled)

dataset["Cluster"] = model.labels_

centers = scaler.inverse_transform(model.cluster_centers_)

centers_df = pd.DataFrame(
    centers,
    columns=X.columns
)

print(dataset)

print("\nCluster Centers:")
print(centers_df.round(2))

print("\nCluster Sizes:")
print(dataset["Cluster"].value_counts().sort_index())

print("\nInertia:", round(model.inertia_, 2))

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=dataset,
    x="Annual_Income",
    y="Total_Purchases",
    hue="Cluster",
    palette="Set1",
    s=100
)

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income")
plt.ylabel("Total Purchases")
plt.legend(title="Customer Cluster")
plt.grid(True, alpha=0.2)

plt.show()