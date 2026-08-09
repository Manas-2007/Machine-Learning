import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    f1_score
)


# =========================================================
# 1. CREATE DATASET
# =========================================================

data = {
    "Message": [
        "Congratulations you won a free prize",
        "Win cash now click this link",
        "Get free lottery tickets today",
        "You have won a free vacation",
        "Claim your free reward now",
        "Congratulations you are selected for a cash prize",
        "Free offer available click now",
        "Win a brand new phone today",
        "Exclusive free deal for you",
        "You won a lottery claim now",

        "Can you send me the assignment",
        "Meeting is scheduled at 10 AM tomorrow",
        "Please call me when you are free",
        "Your project submission is due tomorrow",
        "Can we meet after class",
        "Please share the notes from today's lecture",
        "I will send the report tonight",
        "The exam schedule has been updated",
        "Don't forget to attend the meeting",
        "Can you help me with this problem",

        "Get free money by clicking here",
        "You have been selected to win cash",
        "Claim your free gift immediately",
        "Limited time offer win now",
        "Congratulations claim your prize"
    ],

    "Label": [
        "Spam",
        "Spam",
        "Spam",
        "Spam",
        "Spam",
        "Spam",
        "Spam",
        "Spam",
        "Spam",
        "Spam",

        "Not Spam",
        "Not Spam",
        "Not Spam",
        "Not Spam",
        "Not Spam",
        "Not Spam",
        "Not Spam",
        "Not Spam",
        "Not Spam",
        "Not Spam",

        "Spam",
        "Spam",
        "Spam",
        "Spam",
        "Spam"
    ]
}


dataset = pd.DataFrame(data)


# =========================================================
# 2. INPUT AND OUTPUT
# =========================================================

X = dataset["Message"]
y = dataset["Label"]


# =========================================================
# 3. CONVERT TEXT INTO NUMBERS
# =========================================================

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(X)


# =========================================================
# 4. TRAIN / TEST SPLIT
# =========================================================

x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)


# =========================================================
# 5. CREATE AND TRAIN NAIVE BAYES MODEL
# =========================================================

model = MultinomialNB()

model.fit(x_train, y_train)

print("================== MODEL TRAINED SUCCESSFULLY ==================")


# =========================================================
# 6. PREDICTION ON COMPLETE DATASET
# =========================================================

dataset["Prediction"] = model.predict(X)

print("\n================ AI PREDICTION VS ACTUAL =================")

print(dataset)


# =========================================================
# 7. MODEL EVALUATION
# =========================================================

predictions = model.predict(x_test)


print("\n====================== MODEL EVALUATION ======================")

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test,
    predictions,
    pos_label="Spam"
)

recall = recall_score(
    y_test,
    predictions,
    pos_label="Spam"
)

f1 = f1_score(
    y_test,
    predictions,
    pos_label="Spam"
)


print("Naive Bayes Classification")

print("Accuracy  :", round(accuracy * 100, 2), "%")
print("Precision :", round(precision * 100, 2), "%")
print("Recall    :", round(recall * 100, 2), "%")
print("F1 Score  :", round(f1 * 100, 2), "%")


# =========================================================
# 8. CONFUSION MATRIX
# =========================================================

print("\n====================== CONFUSION MATRIX ======================")

print(
    confusion_matrix(
        y_test,
        predictions,
        labels=["Spam", "Not Spam"]
    )
)


# =========================================================
# 9. TEST WITH A NEW MESSAGE
# =========================================================

new_message = [
    "Congratulations you won a free cash prize"
]

new_message_vector = vectorizer.transform(new_message)

new_prediction = model.predict(new_message_vector)


print("\n====================== NEW MESSAGE TEST ======================")

print("Message    :", new_message[0])
print("Prediction :", new_prediction[0])