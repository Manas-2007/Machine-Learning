import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

dataset = pd.read_excel(
    r"C:\Users\Student\Desktop\Pandas\Machine-Learning\Excel files\DT Regression 1.xlsx"
)

print(dataset)

# ================= FEATURES & TARGET =================

X = dataset[
    [
        "Area_sqft",
        "Bedrooms",
        "Bathrooms",
        "House_Age",
        "Distance_City_km",
        "Nearby_Schools"
    ]
]

y = dataset["Price_Lakh"]

# ================= TRAIN TEST SPLIT =================

x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=42,
    test_size=0.3
)

# ================= RANDOM FOREST =================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(x_train, y_train)

print("=============== MODEL TRAINED SUCCESSFULLY ===========")

# ================= PREDICTION ON FULL DATA =================

dataset["Predicted Prices"] = model.predict(X)

dataset["Predicted Prices"] = round(
    dataset["Predicted Prices"], 2
)

dataset["Error"] = (
    dataset["Price_Lakh"]
    - dataset["Predicted Prices"]
)

print(dataset)

# ================= MODEL EVALUATION =================

y_pred = model.predict(x_test)

print("================= MODEL EVALUATION ==================")

print(
    "Mean Absolute Error :",
    round(mean_absolute_error(y_test, y_pred), 2)
)

print(
    "Mean Squared Error :",
    round(mean_squared_error(y_test, y_pred), 2)
)

print(
    "R2 SCORE :",
    round(r2_score(y_test, y_pred), 2) * 100,
    "%"
)