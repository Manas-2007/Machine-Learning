import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)
from sklearn.preprocessing import StandardScaler

dataset = pd.read_excel(
    r"C:\Users\Student\Desktop\Pandas\Machine-Learning\Excel files\DT Regression 1.xlsx"
)
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

x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

model=KNeighborsRegressor(n_neighbors=7)
model.fit(x_train,y_train)

X_Scaled=scaler.transform(X)
dataset["Predicted Price"]=model.predict(X_Scaled)
dataset["Error"]=dataset["Price_Lakh"]-dataset["Predicted Price"]

print("================ AI VS ACTUAL =================")
print(dataset)


#================ MODEL EVALUATION ===============

y_pred = model.predict(x_test)

print("MAE : ", round(mean_absolute_error(y_test, y_pred), 2))
print("MSE : ", round(mean_squared_error(y_test, y_pred), 2))
print("R2 Score : ", round(r2_score(y_test, y_pred), 2) * 100, "%")