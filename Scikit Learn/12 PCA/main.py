import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt


# Load dataset
dataset = pd.read_csv(
    r"D:\Manas Doc\Pandas\Machine-Learning\Excel files\employee_test.csv"
)


# Select features
X = dataset[
    [
        "Age",
        "Annual_Income",
        "Total_Purchases",
        "Avg_Order_Value",
        "Website_Visits"
    ]
]


# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


# Create PCA dataset
pca_data = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"]
)


# Explained variance
print("Explained Variance Ratio:")
print("PC1:", round(pca.explained_variance_ratio_[0] * 100, 2), "%")
print("PC2:", round(pca.explained_variance_ratio_[1] * 100, 2), "%")

print(
    "Total:",
    round(pca.explained_variance_ratio_.sum() * 100, 2),
    "%"
)


# Visualize PCA
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=pca_data,
    x="PC1",
    y="PC2",
    s=80
)

plt.title("PCA: 5 Dimensions Reduced to 2 Dimensions")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(alpha=0.2)

plt.show()