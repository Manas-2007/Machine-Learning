import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = {
    "Experience": [1, 2, 3, 4, 5, 6, 7, 8],
    "Salary": [22000, 27000, 33000, 41000, 50000, 61000, 73000, 86000]
}

dataset=pd.DataFrame(data)
print("===========ORIGINAL DATASET=============\n",dataset)

# Feature & Target selection (training data)
x=dataset[["Experience"]]
y=dataset["Salary"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=10)

# training the model with data
model=LinearRegression()
model.fit(x_train,y_train)

# Testing with the existing data
dataset["Predicted Salary"]=model.predict(dataset[["Experience"]]).round(1)
dataset["Residual"]=dataset["Predicted Salary"]-dataset["Salary"]
print("\n=============SALARY WITH AI PREDICTION===========\n",dataset)

# user input
user_exp=int(input("Enter Experience ( in years) : "))
user_exp_df=pd.DataFrame({
    "Experience":[user_exp]
})
user_exp_prediction=model.predict(user_exp_df)
print("Predicted Salary ( in lakhs) : ",round(user_exp_prediction[0],1))