import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Probability estimation function
def Status(prediction):
    if(prediction[0]==1):
        return "PASS"
    else:
        return "FAIL"

data = {
    "Hours_Studied": [1,2,3,4,5,6,7,8,9,10],
    "Attendance": [50,55,60,65,70,75,80,85,90,95],
    "Pass": [0,0,0,0,1,1,1,1,1,1]
}
dataset=pd.DataFrame(data)
print("======================= ORIGINAL DATASET ===============================")
print(dataset)

X=dataset[["Hours_Studied","Attendance"]]
y=dataset["Pass"]

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LogisticRegression()
model.fit(x_train,y_train)

dataset["Prediction Pass/Fail"]=model.predict(dataset[["Hours_Studied","Attendance"]])
dataset["Prediction Pass/Fail"]=dataset["Prediction Pass/Fail"].map({
    0:"FAIL",
    1:"PASS"
})



print("\n================= AI PREDICTION =====================\n",dataset)

# BASED ON USER INPUT
hours=int(input("Enter number of study hours : "))
attendance=int(input("Enter total number of attendance : "))
user_info=pd.DataFrame({
    "Hours_Studied":[hours],
    "Attendance":[attendance]
})
y_pred=model.predict(user_info)

print("Expected Final Status : ",Status(y_pred))