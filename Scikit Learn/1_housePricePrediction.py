import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = {
    "House_Size": [500, 700, 900, 1100, 1300, 1500, 1700, 1900],
    "Price": [15, 22, 30, 38, 46, 55, 63, 72]
}
dataset=pd.DataFrame(data)
print("===========Original Dataset========= :\n",dataset)

# Training Data
x=dataset[["House_Size"]]
y=dataset["Price"]

# Train the model
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=10)
model=LinearRegression()
model.fit(x_train,y_train)

# Learned Equation (slope + intercept)
print("\n===== MODEL INFORMATION =====")
print(f"Slope : {model.coef_[0]:.4f}")
print(f"Intercept : {model.intercept_:.4f}")

# Predicting on Existing Dataset
dataset["Predicted_Prices"]=model.predict(dataset[["House_Size"]]).round(1)
dataset["Residual"]=(dataset["Predicted_Prices"]-dataset["Price"])
print("\n========Comparing Existing Prices with Predicted Prices========\n",dataset)

# User Input
house_size=float(input("Enter House Size (sq.ft) : "))

# Convert into dataframe (same format as training x)
new_house=pd.DataFrame({
    "House_Size":[house_size]
})

# AI Prediction
predicted_price=model.predict(new_house)
print("\n============AI PREDICTION============")
print("House Size (in Sq.ft) : ",house_size)
print("Predicted Price : ",round(predicted_price[0],1),"Lakhs")