import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (LabelEncoder,StandardScaler)
from sklearn.metrics import (confusion_matrix,f1_score,recall_score,accuracy_score,precision_score)
from sklearn.neighbors import KNeighborsClassifier

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

label=LabelEncoder()
dataset["Placement_Status"]=label.fit_transform(dataset["Placement_Status"])
y=dataset["Placement_Status"]
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

