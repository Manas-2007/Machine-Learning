import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = {
    "RAM_GB": [4, 8, 8, 16, 16, 32, 32, 64, 16, 8],
    "Storage_GB": [256, 256, 512, 512, 1024, 1024, 2048, 2048, 512, 256],
    "Processor_Gen": [8, 10, 11, 11, 12, 13, 13, 14, 12, 10],
    "Battery_Hours": [5, 6, 7, 8, 9, 10, 11, 12, 8, 6],
    "Price": [35000, 48000, 56000, 72000, 85000, 115000, 145000, 210000, 78000, 50000]
}
dataset=pd.DataFrame(data)
print("=========== ORIGINAL DATASET =============\n",dataset)

# Features & Target selection 
x=dataset[["RAM_GB","Storage_GB","Processor_Gen","Battery_Hours"]]
y=dataset["Price"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)

# Model Training
model=LinearRegression()
model.fit(x_train,y_train)

# Slops & Intercept
print("Slopes (Coeffients) : ",model.coef_)
print("Intercept : ",round(model.intercept_,1))

# Prediction Existing Prices
dataset["Predicted Prices"]=model.predict(dataset[["RAM_GB","Storage_GB","Processor_Gen","Battery_Hours"]]).round(1)
dataset["Residual"]=dataset["Price"]-dataset["Predicted Prices"]
print("\n============== AI PREDICTION ================\n",dataset)

# User Action
ram_size=int(input("Enter RAM size : "))
storage=int(input("Enter Storage : "))
processor_gen=int(input("Enter Processor Generation : "))
battery_size=int(input("Enter Battery Backup : "))

user_Data=pd.DataFrame({
     "RAM_GB": [ram_size],
        "Storage_GB": [storage],
        "Processor_Gen": [processor_gen],
        "Battery_Hours": [battery_size]
})
user_prediction=model.predict(user_Data)
print("Predicted Price : ",round(user_prediction[0],1))