import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    confusion_matrix,
    recall_score
)


dataset = pd.read_excel(
    r"C:\Users\Student\Desktop\Pandas\Machine-Learning\Excel files\DT Classification 2.xlsx"
)


X = dataset[[
    "CGPA",
    "DSA_Problems",
    "Projects",
    "Internship",
    "Aptitude",
    "Communication",
    "Attendance",
    "Hackathons"
]]


# Convert output labels into numbers
label = LabelEncoder()

dataset["Placement_Status"] = label.fit_transform(
    dataset["Placement_Status"]
)

y = dataset["Placement_Status"]


# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)


# ================= SCALING =================

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# ================= KNN MODEL =================

model = KNeighborsClassifier(n_neighbors=5)

model.fit(x_train, y_train)

print("=============== KNN MODEL TRAINED SUCCESSFULLY ===============")


# ================= PREDICTION =================

# Scale complete dataset before prediction
X_scaled = scaler.transform(X)

dataset["Predicted Status"] = model.predict(X_scaled)


# Convert numbers back to original labels
dataset["Placement_Status"] = label.inverse_transform(
    dataset["Placement_Status"]
)

dataset["Predicted Status"] = label.inverse_transform(
    dataset["Predicted Status"]
)


print("================ AI PREDICTION VS ACTUAL RESULTS ======================")
print(dataset)


# ================= MODEL EVALUATION =================

y_pred = model.predict(x_test)


print("\n========================= MODEL EVALUATION =========================")

print("----- KNN CLASSIFICATION ---------")

print(
    "Accuracy :",
    round(accuracy_score(y_test, y_pred), 2) * 100,
    "%"
)

print(
    "Precision :",
    round(precision_score(y_test, y_pred), 2) * 100,
    "%"
)

print(
    "Recall :",
    round(recall_score(y_test, y_pred), 2) * 100,
    "%"
)

print(
    "F1 Score :",
    round(f1_score(y_test, y_pred), 2) * 100,
    "%"
)


print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))