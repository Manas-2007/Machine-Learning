import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
    
dataset=pd.read_excel(r"C:\Users\Student\Desktop\Pandas\Machine-Learning\Excel files\Loan Approval.xlsx")
encoder=LabelEncoder()

X=dataset[["Age","Annual_Income","Credit_Score","Existing_Loans","Employment_Years"]]

# Label -> Numbers (label encoding)
dataset["Loan_Status"]=encoder.fit_transform(dataset["Loan_Status"])
y=dataset["Loan_Status"]
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,randoṇm_state=42)

model=LogisticRegression()
model.fit(x_train,y_train)


dataset["Predicted Status"]=model.predict(X)

# Label decoding (numbers to text back)
dataset["Predicted Status"]=encoder.inverse_transform(dataset["Predicted Status"])

print("============================== AI PREDICTION VS ACTUAL DATASET ============================")
print(dataset)

# User input
age=int(input("Enter age : "))
income=int(input("Enter annual income : "))
credit=int(input("Enter credit score : "))
loans=int(input("Enter existing loans : "))
employ_yrs=int(input("Enter total employment years : "))

user_info=pd.DataFrame({
    "Age":[age],
    "Annual_Income" :[income],
    "Credit_Score":[credit],
    "Existing_Loans":[loans],
    "Employment_Years":[employ_yrs]
})

model_pred=model.predict(user_info)
model_pred=encoder.inverse_transform(model_pred)
print("Expected Loan Status : ",model_pred[0])

