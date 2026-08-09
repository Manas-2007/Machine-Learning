import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # <--- Naya Import
from sklearn.metrics import accuracy_score
import pickle
import warnings

warnings.filterwarnings("ignore")

print("1. Loading Smart Dataset...")
df = pd.read_csv("smart_robot_dataset.csv")

# X (Features) aur Y (Target) alag kar rahe hain
X = df[['Left_Dist', 'Center_Dist', 'Right_Dist']]
Y = df['Action_Y']

print("2. Splitting data into Training and Testing sets...")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("3. Training the Random Forest Model...")
# n_estimators=100 matlab hum 100 trees ka jungle bana rahe hain!
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, Y_train)

print("4. Evaluating Model (Taking the Exam)...")
predictions = model.predict(X_test)
accuracy = accuracy_score(Y_test, predictions)
print(f"--> AI Model Accuracy: {accuracy * 100:.2f}%\n")

print("5. Saving the Smart Brain...")
with open('smart_brain.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Success! 'smart_brain.pkl' is ready for deployment. 🚀")