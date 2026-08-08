import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,precision_score,f1_score,confusion_matrix,recall_score)

dataset=pd.read_excel(r"C:\Users\Student\Desktop\Pandas\Machine-Learning\Excel files\DT Classification 2.xlsx")

X = dataset[[
    "CGPA",
    "DSA_Problems",
    "Projects",
    "Internship",
    "Aptitude",
    "Communication",
    "Attendance",
    "Hackathons"
]]

label = LabelEncoder()
dataset["Placement_Status"] = label.fit_transform(dataset["Placement_Status"])
y = dataset["Placement_Status"]
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

model=RandomForestClassifier(random_state=42,n_estimators=100)
model.fit(x_train,y_train)

dataset["Predicted Status"]=model.predict(X)
dataset["Placement_Status"]=label.inverse_transform(dataset["Placement_Status"])
dataset["Predicted Status"]=label.inverse_transform(dataset["Predicted Status"])

print("================ AI PREDICTION VS ACTUAL RESULTS ======================")
print(dataset)

print("\n========================= MODEL EVALUATION =========================")
y_pred1=model.predict(x_test)
print("----- RANDOM FOREST ALGORITHM ---------")
print("Accuracy :", round(accuracy_score(y_test, y_pred1), 2) * 100, "%")
print("Precision :", round(precision_score(y_test, y_pred1), 2) * 100, "%")
print("Recall :", round(recall_score(y_test, y_pred1), 2) * 100, "%")
print("F1 Score :", round(f1_score(y_test, y_pred1), 2) * 100, "%")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred1))
