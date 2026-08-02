import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)
import joblib

data = {
    "Age": [19, 22, 28, 35, 42, 50, 58, 63, 30, 40],
    "BMI": [21.5, 24.3, 27.8, 30.1, 31.5, 29.2, 33.4, 35.0, 26.0, 28.5],
    "Children": [0, 1, 0, 2, 3, 2, 4, 3, 1, 2],
    "Smoker": [0, 0, 1, 0, 1, 0, 1, 1, 0, 1],
    "Insurance_Cost": [1800, 2200, 5200, 3500, 7800, 4800, 9800, 12000, 3000, 6900]
}

dataset=pd.DataFrame(data)
print("============== ORIGINAL DATA =================\n",dataset)

# Features and Target selection
x=dataset[["Age","BMI","Children","Smoker"]]
y=dataset["Insurance_Cost"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# Model Learning
model=LinearRegression()
model.fit(x_train,y_train)

# Saving the ai model
joblib.dump(model,"insurance_model.pkl")

# Intercept and slops
print("Slope (coefficients) : ",model.coef_)
print("Intercept : ",round(model.intercept_,1))

# Prediction Test on Existing Prices
dataset["Predicted Prices"]=model.predict(dataset[["Age","BMI","Children","Smoker"]]).round(1)
dataset["Residual"]=dataset["Insurance_Cost"]-dataset["Predicted Prices"]
print("\n===================== AI PREDICTION ==========================")
print(dataset)

# User test
age=int(input("Enter age  :"))
bmi=float(input("Enter BMI value :"))
children=int(input("Enter Number of children : "))
smoker=int(input("Smoker (YES = 1 / NO = 0) :")) 
user_data=pd.DataFrame({
    "Age": [age],
    "BMI": [bmi],
    "Children": [children],
    "Smoker": [smoker]
})
insurance_prediction=model.predict(user_data)
print("Expected Insurance price (in Rupees) : ",round(insurance_prediction[0],1))

# Model evaluation
y_pred=model.predict(x_test)
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=mse**0.5
r2=r2_score(y_test,y_pred)

print("\n============ MODEL EVALUATION ================")
print(f"MAE : {mae:.2f}")
print(f"MSE : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2 : {r2:.4f}")
