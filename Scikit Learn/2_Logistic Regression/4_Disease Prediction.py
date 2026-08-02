import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

dataset=pd.read_excel(r"C:\Users\Student\Desktop\Disease Prediction.xlsx")
print("=============================== ORIGINAL DATASET ================================\n",dataset)

# Converting original data into numbers (label encoding)
encoder=LabelEncoder()
dataset["Disease"]=encoder.fit_transform(dataset["Disease"])

X=dataset[["Age","Temperature_C","BP_Systolic","Cough","Body_Pain","WBC_Count"]]
y=dataset["Disease"]
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LogisticRegression()
model.fit(x_train,y_train)
prediction = model.predict(X)

dataset["Predicted Disease"] = encoder.inverse_transform(prediction)
print("\n===================== AI DATASET ==========================\n",dataset)