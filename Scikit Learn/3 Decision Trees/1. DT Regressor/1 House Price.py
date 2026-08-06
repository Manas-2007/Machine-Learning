import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)

dataset = pd.DataFrame({
    "Area_sqft":[
        820,1500,980,2300,1200,
        1750,900,2100,1350,2600,
        780,3200,1600,1450,1950,
        1100,2800,1700,1000,2400,
        1550,890,2050,1300,1850
    ],

    "Bedrooms":[
        2,3,2,5,3,
        4,2,5,3,6,
        1,5,4,3,4,
        2,6,4,2,5,
        3,2,5,3,4
    ],

    "Bathrooms":[
        1,2,2,4,2,
        3,1,4,2,5,
        1,5,3,2,3,
        2,5,3,2,4,
        2,1,4,2,3
    ],

    "House_Age":[
        18,4,10,2,16,
        8,22,1,11,0,
        27,3,5,13,6,
        20,1,7,19,2,
        9,25,3,15,5
    ],

    "Distance_City_km":[
        16,5,11,18,8,
        6,20,4,12,22,
        15,7,9,3,13,
        19,2,17,10,6,
        8,23,5,14,9
    ],

    "Nearby_Schools":[
        2,8,5,3,7,
        6,4,9,5,2,
        3,8,7,9,5,
        4,6,3,8,5,
        7,2,9,4,6
    ],

    "Price_Lakh":[
        34,
        98,
        42,
        120,
        58,
        87,
        29,
        132,
        64,
        141,
        21,
        168,
        81,
        73,
        92,
        48,
        158,
        85,
        39,
        126,
        77,
        25,
        118,
        69,
        96
    ]}
)

X=dataset[["Area_sqft","Bedrooms","Bathrooms","House_Age","Distance_City_km","Nearby_Schools"]]
y=dataset["Price_Lakh"]
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

# Model Training and Prediction
model=DecisionTreeRegressor()
model.fit(x_train,y_train)

# Testing on Existing Data
dataset["Predicted_Price_Lakh"]=model.predict(X)
print("========================= DATASET WITH PREDICTED PRICES =========================")
print(dataset)

# Model Evaluation
y_pred=model.predict(x_test)
print("\n========================= MODEL EVALUATION =========================")
print("Mean Absolute Error :",round(mean_absolute_error(y_test,y_pred),2))
print("Mean Squared Error :",round(mean_squared_error(y_test,y_pred),2))
print("R2 Score :",round(r2_score(y_test,y_pred),2)*100,"%")
