import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,confusion_matrix)
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

dataset=pd.read_excel(r"C:\Users\Student\Desktop\Pandas\Machine-Learning\Excel files\DT Classification 2.xlsx")
print(dataset)

X=dataset[["CGPA","DSA_Problems","Projects","Internship","Aptitude","Communication","Attendance","Hackathons"]]

label=LabelEncoder()
dataset["Placement_Status"]=label.fit_transform(dataset["Placement_Status"])
y=dataset["Placement_Status"]
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=XGBClassifier(n_estimators=100,max_depth=3,learning_rate=0.0002,random_state=42)
model.fit(x_train,y_train)
print("================== MODEL TRAINED SUCCESSFULLY USING XGBCLASSIFIER =============================")

predictions=model.predict(x_test)
print("\n====================== MODEL EVALUATION ======================")

print("XGBoost Classification")

print("Accuracy  :", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("Precision :", round(precision_score(y_test, predictions) * 100, 2), "%")
print("Recall    :", round(recall_score(y_test, predictions) * 100, 2), "%")
print("F1 Score  :", round(f1_score(y_test, predictions) * 100, 2), "%")

print("\n====================== CONFUSION MATRIX ======================")
print(confusion_matrix(y_test, predictions))