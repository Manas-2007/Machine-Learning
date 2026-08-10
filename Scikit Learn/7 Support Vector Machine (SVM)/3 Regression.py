import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================================================
# 1. CREATE DATASET
# =========================================================

data = {
    "CGPA": [
        6.1, 6.3, 6.5, 6.7, 6.8,
        7.0, 7.1, 7.2, 7.3, 7.4,
        7.5, 7.6, 7.7, 7.8, 7.9,
        8.0, 8.1, 8.2, 8.3, 8.4,
        8.5, 8.6, 8.7, 8.8, 8.9,
        9.0, 9.1, 9.2, 9.3, 9.4,
        6.2, 6.4, 6.6, 6.9, 7.15,
        7.25, 7.35, 7.45, 7.55, 7.65,
        7.75, 7.85, 8.15, 8.25, 8.35,
        8.45, 8.55, 8.75, 8.95, 9.5
    ],

    "Internships": [
        0, 0, 1, 0, 1,
        1, 0, 1, 1, 0,
        1, 1, 0, 2, 1,
        2, 1, 2, 1, 2,
        2, 2, 1, 2, 3,
        2, 3, 2, 3, 3,
        3, 1, 2, 1, 2,
        0, 2, 1, 2, 1,
        3, 2, 3, 2, 3,
        3, 2, 3, 3, 4
    ],

    "Salary_LPA": [
        3.2, 3.4, 3.8, 3.6, 4.0,
        4.2, 4.1, 4.5, 4.7, 4.6,
        5.0, 5.2, 5.1, 5.5, 5.7,
        6.0, 6.2, 6.4, 6.6, 6.8,
        7.1, 7.3, 7.5, 7.8, 8.0,
        8.3, 8.6, 8.8, 9.1, 9.4,
        3.3, 3.7, 3.9, 4.2, 4.4,
        4.8, 4.9, 5.3, 5.5, 5.8,
        6.1, 6.5, 6.7, 7.0, 7.2,
        7.6, 8.1, 8.5, 9.0, 10.2
    ]
}

dataset = pd.DataFrame(data)

print("========================= DATASET =========================")
print(dataset)


# =========================================================
# 2. SEPARATE FEATURES AND TARGET
# =========================================================

X = dataset[["CGPA", "Internships"]]

# Continuous target → Regression
y = dataset["Salary_LPA"]


# =========================================================
# 3. TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================================================
# 4. CREATE SVR MODEL
# =========================================================

# StandardScaler:
# Features ko same scale par laata hai.
#
# SVR:
# Support Vector Regression

model = make_pipeline(
    StandardScaler(),
    SVR(
        kernel="rbf",
        C=10,
        gamma="scale",
        epsilon=0.1
    )
)


# =========================================================
# 5. TRAIN MODEL
# =========================================================

model.fit(X_train, y_train)

print("\n================ MODEL TRAINED SUCCESSFULLY ================")


# =========================================================
# 6. PREDICTION
# =========================================================

predictions = model.predict(X_test)


# =========================================================
# 7. MODEL EVALUATION
# =========================================================

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\n====================== MODEL EVALUATION ======================")

print("SVR REGRESSION")

print(f"MAE  : {mae:.2f} LPA")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f} LPA")
print(f"R² Score : {r2 * 100:.2f} %")


# =========================================================
# 8. ACTUAL VS PREDICTED
# =========================================================

comparison = pd.DataFrame({
    "Actual Salary": y_test.values,
    "Predicted Salary": predictions
})

print("\n====================== ACTUAL VS PREDICTED =================")

print(comparison)


# =========================================================
# 9. TEST A NEW STUDENT
# =========================================================

new_student = pd.DataFrame({
    "CGPA": [8.3],
    "Internships": [2]
})

new_prediction = model.predict(new_student)

print("\n====================== NEW STUDENT TEST =====================")

print("CGPA          : 8.3")
print("Internships   : 2")
print(f"Predicted Salary : {new_prediction[0]:.2f} LPA")


# =========================================================
# 10. VISUALIZATION
# =========================================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=dataset,
    x="CGPA",
    y="Salary_LPA",
    s=100
)

# Sort CGPA so the prediction curve is smooth
sorted_data = dataset.sort_values("CGPA")

X_curve = sorted_data[["CGPA", "Internships"]]

y_curve = model.predict(X_curve)

plt.plot(
    sorted_data["CGPA"],
    y_curve,
    linewidth=2,
    label="SVR Prediction"
)

plt.title("SVR Regression - Student Salary Prediction")
plt.xlabel("CGPA")
plt.ylabel("Salary (LPA)")
plt.legend()
plt.grid(True)

plt.show()