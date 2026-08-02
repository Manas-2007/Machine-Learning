import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def get_fraud_status(prediction):
    if prediction[0]==0:
        return "Legitimate"
    else:
        return "Fraud"
    
dataset=pd.read_excel(r"C:\Users\Student\Desktop\customer_status.xlsx")
print("====================== ORIGINAL DATASET ========================\n",dataset)

X=dataset[["Transaction_Amount","Location_Risk","Device_Trust","Transactions_Today"]]
y=dataset["Fraud"]
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LogisticRegression()
model.fit(x_train,y_train)

dataset["Fraud Prediction"]=model.predict(dataset[["Transaction_Amount","Location_Risk","Device_Trust","Transactions_Today"]])
fraud_label={
    0:"Legitimate",
    1:"Fraud"
}
dataset["Fraud Prediction"]=dataset["Fraud Prediction"].map(fraud_label)
print("=================== AI PREDICTION =======================\n",dataset)

# user test
amount=int(input("Enter transaction amount :"))
location=int(input("Enter Location Risk (0/1) :"))
device=int(input("Enter Device Trust (0/1) :"))
transaction_today=int(input("Enter Today's Transactions :"))
user_info=pd.DataFrame({
    "Transaction_Amount" :[amount],
    "Location_Risk":[location],
    "Device_Trust":[device],
    "Transactions_Today":[transaction_today]
})
y_pred=model.predict(user_info)
print("Transaction Status : ",get_fraud_status(y_pred))
