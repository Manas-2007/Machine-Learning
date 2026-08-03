import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)

dataset=pd.read_excel(r"C:\Users\Student\Desktop\Pandas\Machine-Learning\Excel files\Car Price Prediction.xlsx")
print("========================== ORIGINAL DATASET ==============================\n",dataset)

X=dataset[["Car_Age","KM_Driven","Engine_CC","Owners"]]
y=dataset["Selling_Price"]
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(x_train,y_train)

dataset["Predicted Prices"]=model.predict(X).round(1)
dataset["Error"]=dataset["Selling_Price"]-dataset["Predicted Prices"]

print("\n================= AI PREDICTION ================\n",dataset)

# User input
car_age=int(input("Enter the car age : "))
km_driven=int(input("Enter Previous KM reading : "))
engine_CC=int(input("Engine Power (in CC )"))
prev_owners=int(input("Previous Number of Owners :" ))

user_data=pd.DataFrame({
    "Car_Age":[car_age],
    "KM_Driven":[km_driven],
    "Engine_CC":[engine_CC],
    "Owners":[prev_owners]
})
model_pred=model.predict(user_data)
print("Estimated Selling Price (in Rupees) : ",round(model_pred[0],0))

# model evaluation
y_pred=model.predict(x_test)
print("Mean Absolute Error : ",round(mean_absolute_error(y_test,y_pred),1))
print("Mean Squared Error : ",round(mean_squared_error(y_test,y_pred),1))
print("R2 Score : ",round(r2_score(y_test,y_pred),4))