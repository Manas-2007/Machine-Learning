import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# =========================================================
# 1. DATASET
# =========================================================

data = {

    "CGPA": [
        6.1, 6.4, 6.8, 7.0, 7.2,
        7.4, 7.6, 7.8, 8.0, 8.1,
        8.2, 8.3, 8.4, 8.5, 8.6,
        8.7, 8.8, 8.9, 9.0, 9.1,
        9.2, 9.3, 9.4, 7.1, 8.9
    ],

    "DSA_Problems": [
        90, 180, 130, 420, 210,
        300, 150, 480, 270, 190,
        520, 330, 610, 290, 450,
        350, 700, 410, 230, 550,
        380, 620, 310, 560, 160
    ],

    "Placement_Status": [
        "Not Placed",
        "Not Placed",
        "Not Placed",
        "Placed",
        "Not Placed",

        "Placed",
        "Not Placed",
        "Placed",
        "Not Placed",
        "Not Placed",

        "Placed",
        "Not Placed",
        "Placed",
        "Placed",
        "Placed",

        "Not Placed",
        "Placed",
        "Placed",
        "Placed",
        "Not Placed",

        "Placed",
        "Placed",
        "Not Placed",
        "Not Placed",
        "Placed"
    ]
}


dataset = pd.DataFrame(data)


print("========================= DATASET =========================")

print(dataset)


# =========================================================
# 2. INPUT AND OUTPUT
# =========================================================

X = dataset[
    [
        "CGPA",
        "DSA_Problems"
    ]
]

y = dataset["Placement_Status"]


# =========================================================
# 3. TRAIN / TEST SPLIT
# =========================================================

x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================================================
# 4. CREATE SVM MODEL
# =========================================================

model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale"
)


# =========================================================
# 5. TRAIN MODEL
# =========================================================

model.fit(
    x_train,
    y_train
)


print("\n================ MODEL TRAINED SUCCESSFULLY ================")


# =========================================================
# 6. PREDICTION
# =========================================================

predictions = model.predict(
    x_test
)


# =========================================================
# 7. MODEL EVALUATION
# =========================================================

print("\n====================== MODEL EVALUATION ======================")

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    pos_label="Placed",
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    pos_label="Placed",
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    pos_label="Placed",
    zero_division=0
)


print("SVM CLASSIFICATION")

print("Accuracy  :", round(accuracy * 100, 2), "%")
print("Precision :", round(precision * 100, 2), "%")
print("Recall    :", round(recall * 100, 2), "%")
print("F1 Score  :", round(f1 * 100, 2), "%")


# =========================================================
# 8. CONFUSION MATRIX
# =========================================================

print("\n====================== CONFUSION MATRIX ======================")

cm = confusion_matrix(
    y_test,
    predictions,
    labels=[
        "Not Placed",
        "Placed"
    ]
)

print(cm)


# =========================================================
# 9. ACTUAL VS PREDICTED
# =========================================================

result = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})


print("\n================ ACTUAL VS PREDICTED =================")

print(result)


# =========================================================
# 10. NEW STUDENT TEST
# =========================================================

new_student = pd.DataFrame({
    "CGPA": [7.7],
    "DSA_Problems": [350]
})


new_prediction = model.predict(
    new_student
)


print("\n====================== NEW STUDENT TEST ======================")

print(
    "CGPA          :",
    new_student["CGPA"][0]
)

print(
    "DSA Problems  :",
    new_student["DSA_Problems"][0]
)

print(
    "Prediction    :",
    new_prediction[0]
)


# =========================================================
# 11. SVM DECISION BOUNDARY
# =========================================================

plt.figure(
    figsize=(10, 7)
)


# Plot actual dataset

sns.scatterplot(
    data=dataset,
    x="CGPA",
    y="DSA_Problems",
    hue="Placement_Status",
    style="Placement_Status",
    s=100
)


# =========================================================
# 12. CREATE GRID
# =========================================================

x_min = dataset["CGPA"].min() - 0.2
x_max = dataset["CGPA"].max() + 0.2

y_min = dataset["DSA_Problems"].min() - 30
y_max = dataset["DSA_Problems"].max() + 30


xx, yy = np.meshgrid(
    np.linspace(
        x_min,
        x_max,
        300
    ),
    np.linspace(
        y_min,
        y_max,
        300
    )
)


# =========================================================
# 13. PREDICT GRID POINTS
# =========================================================

grid = pd.DataFrame({
    "CGPA": xx.ravel(),
    "DSA_Problems": yy.ravel()
})


Z = model.predict(
    grid
)


# Convert classes into numbers

Z = np.where(
    Z == "Placed",
    1,
    0
)


Z = Z.reshape(
    xx.shape
)


# =========================================================
# 14. DRAW DECISION BOUNDARY
# =========================================================

plt.contour(
    xx,
    yy,
    Z,
    levels=[0.5],
    linewidths=2
)


# =========================================================
# 15. GRAPH SETTINGS
# =========================================================

plt.title(
    "SVM Classification - Noisy Placement Dataset",
    fontsize=16
)

plt.xlabel(
    "CGPA",
    fontsize=12
)

plt.ylabel(
    "DSA Problems Solved",
    fontsize=12
)

plt.grid(
    True
)

plt.show()