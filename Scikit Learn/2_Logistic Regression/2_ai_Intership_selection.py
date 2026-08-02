import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def get_status(prediction):
    if prediction[0]==1:
        return "SELECTED"
    else:
        return "REJECTED"


dataset=pd.read_excel(r"C:\Users\Student\Desktop\customer_status.xlsx")
print("===================== ORIGINAL DATASET ======================\n",dataset)

X=dataset[["CGPA","DSA_Score","Projects","Communication"]]
y=dataset["Selected"]
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LogisticRegression()
model.fit(x_train,y_train)

dataset["Prediction Status"]=model.predict(dataset[["CGPA","DSA_Score","Projects","Communication"]])
selection_label={
    1:"SELECTED",
    0:"REJECTED"
}
dataset["Prediction Status"]=dataset["Prediction Status"].map(selection_label)
print("\n================ AI PREDICTION ======================\n",dataset)

# User Input
cgpa=float(input("Enter your CGPA : "))
dsa_score=int(input("Enter your DSA score : "))
communication=int(input("Enter your soft skill score : "))
projects=int(input("Enter your number of projects : "))

user_info=pd.DataFrame({
    "CGPA":[cgpa],
    "DSA_Score":[dsa_score],
    "Projects":[projects],
    "Communication" :[communication]
})
y_pred=model.predict(user_info)
print("Estimated Application Status : ",get_status(y_pred))