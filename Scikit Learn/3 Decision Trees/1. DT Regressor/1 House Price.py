import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)

dataset=pd.read_excel(r"C:\Users\Student\Desktop\Pandas\Machine-Learning\Excel files\DT Regression 1.xlsx")
X=dataset[["Area_sqft","Bedrooms","Bathrooms","House_Age","Distance_City_km","Nearby_Schools"]]
y=dataset["Price_Lakh"]
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

# Model Training and Prediction
model=DecisionTreeRegressor(random_state=42)
model2=LinearRegression()
model.fit(x_train,y_train)
model2.fit(x_train,y_train)

# Testing on Existing Data
dataset["Predicted_Price_Lakh"]=model.predict(X)
dataset["Predicted_Price_Lakh_2"]=model2.predict(X)
print("========================= DATASET WITH PREDICTED PRICES =========================")
print(dataset)

# Model Evaluation
y_pred=model.predict(x_test)
y_pred2=model2.predict(x_test)
print("\n========================= MODEL EVALUATION =========================")
print("--------DECISION TREE --------")
print("Mean Absolute Error :",round(mean_absolute_error(y_test,y_pred),2))
print("Mean Squared Error :",round(mean_squared_error(y_test,y_pred),2))
print("R2 Score :",round(r2_score(y_test,y_pred),2)*100,"%")

print("--------LINEAR REGRESSION -------------")
print("Mean Absolute Error :",round(mean_absolute_error(y_test,y_pred2),2))
print("Mean Squared Error :",round(mean_squared_error(y_test,y_pred2),2))
print("R2 Score :",round(r2_score(y_test,y_pred2),2)*100,"%")
