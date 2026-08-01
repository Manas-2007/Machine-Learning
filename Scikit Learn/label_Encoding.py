import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)

encoder=LabelEncoder()
data = {
    "Experience": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Education": [
        "B.Tech",
        "B.Tech",
        "M.Tech",
        "B.Tech",
        "MBA",
        "M.Tech",
        "MBA",
        "PhD",
        "PhD",
        "MBA"
    ],
    "Department": [
        "IT",
        "IT",
        "IT",
        "HR",
        "Finance",
        "Finance",
        "HR",
        "Research",
        "Research",
        "Finance"
    ],
    "Remote_Work": [
        "Yes",
        "No",
        "Yes",
        "No",
        "Yes",
        "No",
        "Yes",
        "No",
        "Yes",
        "No"
    ],
    "Salary": [
        25000,
        30000,
        42000,
        38000,
        55000,
        68000,
        62000,
        90000,
        98000,
        85000
    ]
}
dataset=pd.DataFrame(data)
print("============ ORIGINAL DATASET =============\n",dataset)

education_encoder=LabelEncoder()
department_encoder=LabelEncoder()
remote_work_encoder=LabelEncoder()
dataset["Education"]=education_encoder.fit_transform(dataset["Education"])
dataset["Department"]=department_encoder.fit_transform(dataset["Department"])
dataset["Remote_Work"]=remote_work_encoder.fit_transform(dataset["Remote_Work"])


X=dataset[["Experience","Education","Department","Remote_Work"]]
y=dataset["Salary"]
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


model=LinearRegression()
model.fit(x_train,y_train)

dataset["Predicted Salary"]=model.predict(dataset[["Experience","Education","Department","Remote_Work"]]).round(0)
dataset["Errors"]=dataset["Salary"]-dataset["Predicted Salary"]
print("\n=================== AI PREDICTION SYSTEM ===================")
print(dataset)

# model evaluation
y_prediction=model.predict(x_test)
mae=mean_absolute_error(y_test,y_prediction)
mean_squared_error=mean_squared_error(y_test,y_prediction)
rmse=mean_squared_error**0.5
r2=r2_score(y_test,y_prediction)
print("\n========== MODEL EVALUATION ==========")
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mean_squared_error:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")
