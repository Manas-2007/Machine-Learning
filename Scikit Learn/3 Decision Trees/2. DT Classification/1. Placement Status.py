import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
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

linear_model=LogisticRegression(max_iter=1000)
tree_model=DecisionTreeClassifier(random_state=42)

linear_model.fit(x_train,y_train)
tree_model.fit(x_train,y_train)

dataset["Linear Status"]=linear_model.predict(X)
dataset["Tree Status"]=tree_model.predict(X)

dataset["Tree Status"]=label.inverse_transform(dataset["Tree Status"])
dataset["Linear Status"]=label.inverse_transform(dataset["Linear Status"])
dataset["Placement_Status"]=label.inverse_transform(dataset["Placement_Status"])

print("================ AI PREDICTION VS ACTUAL RESULTS ======================")
print(dataset)

# Prediction on test data
y_pred1 = linear_model.predict(x_test)
y_pred2=tree_model.predict(x_test)

print("\n========================= MODEL EVALUATION =========================")
print("----- LOGISTIC REGRESSION ---------")
print("Accuracy :", round(accuracy_score(y_test, y_pred1), 2) * 100, "%")
print("Precision :", round(precision_score(y_test, y_pred1), 2) * 100, "%")
print("Recall :", round(recall_score(y_test, y_pred1), 2) * 100, "%")
print("F1 Score :", round(f1_score(y_test, y_pred1), 2) * 100, "%")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred1))

print("----- DECISION TREE REGRESSION ---------")
print("Accuracy :", round(accuracy_score(y_test, y_pred2), 2) * 100, "%")
print("Precision :", round(precision_score(y_test, y_pred2), 2) * 100, "%")
print("Recall :", round(recall_score(y_test, y_pred2), 2) * 100, "%")
print("F1 Score :", round(f1_score(y_test, y_pred2), 2) * 100, "%")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred2))

