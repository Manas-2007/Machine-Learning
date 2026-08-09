import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =========================================================
# 1. CREATE DATASET
# =========================================================

data = {
    "Title": [
        "Stock market rises today",
        "Government announces new policy",
        "Team wins championship",
        "New smartphone launched",
        "Company reports strong profit",
        "Central bank cuts interest rates",
        "Football team loses final",
        "New AI model released",
        "Technology stocks gain",
        "Government increases taxes",

        "Player scores winning goal",
        "New laptop receives positive reviews",
        "Bank announces new loan scheme",
        "Company faces major loss",
        "AI startup raises funding",
        "Cricket team wins series",
        "New electric car launched",
        "Interest rates expected to rise",
        "Technology company launches product",
        "Government announces economic reforms",

        "Stock prices fall sharply",
        "Football player injured",
        "New smartphone receives poor reviews",
        "Company profits increase",
        "AI technology improves business"
    ],

    "Description": [
        "Investors are optimistic as major stocks move higher",
        "Officials introduced a new economic policy for businesses",
        "The football team defeated its opponent in the final match",
        "The company introduced its latest smartphone with improved features",
        "Quarterly earnings show strong growth in company revenue",
        "The central bank reduced rates to support economic growth",
        "The team suffered a disappointing defeat in the final",
        "Researchers introduced a powerful artificial intelligence model",
        "Major technology companies recorded gains in the stock market",
        "The government announced higher taxes for several industries",

        "The striker scored in the final minutes to secure victory",
        "Customers praised the performance and battery life of the laptop",
        "The bank introduced a new loan product for customers",
        "The company reported declining revenue and financial problems",
        "The startup received investment from several technology investors",
        "The cricket team defeated its opponent in the final match",
        "The manufacturer unveiled a new electric vehicle",
        "Economists expect borrowing costs to increase next month",
        "The technology company announced its newest software product",
        "New reforms are expected to affect businesses and investors",

        "Investors sold shares as concerns about the economy increased",
        "The football player suffered an injury during the match",
        "Users complained about the smartphone's performance",
        "The company reported higher earnings this quarter",
        "Businesses are adopting artificial intelligence to improve productivity"
    ],

    "Category": [
        "Business",
        "Politics",
        "Sports",
        "Technology",
        "Business",
        "Business",
        "Sports",
        "Technology",
        "Business",
        "Politics",

        "Sports",
        "Technology",
        "Business",
        "Business",
        "Technology",
        "Sports",
        "Technology",
        "Business",
        "Technology",
        "Politics",

        "Business",
        "Sports",
        "Technology",
        "Business",
        "Technology"
    ]
}


dataset = pd.DataFrame(data)


print("========================= DATASET =========================")

print(dataset)


# =========================================================
# 2. COMBINE TEXT FEATURES
# =========================================================

dataset["Text"] = (
    dataset["Title"] + " " + dataset["Description"]
)


# =========================================================
# 3. INPUT AND OUTPUT
# =========================================================

X = dataset["Text"]

y = dataset["Category"]


# =========================================================
# 4. CONVERT TEXT INTO NUMBERS
# =========================================================

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(X)


# =========================================================
# 5. TRAIN / TEST SPLIT
# =========================================================

x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================================================
# 6. CREATE NAIVE BAYES MODEL
# =========================================================

model = MultinomialNB()


# =========================================================
# 7. TRAIN MODEL
# =========================================================

model.fit(x_train, y_train)


print("\n================ MODEL TRAINED SUCCESSFULLY ================")


# =========================================================
# 8. PREDICTION ON TEST DATA
# =========================================================

predictions = model.predict(x_test)


# =========================================================
# 9. MODEL EVALUATION
# =========================================================

print("\n====================== MODEL EVALUATION ======================")

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)


print("Naive Bayes Classification")

print("Accuracy  :", round(accuracy * 100, 2), "%")
print("Precision :", round(precision * 100, 2), "%")
print("Recall    :", round(recall * 100, 2), "%")
print("F1 Score  :", round(f1 * 100, 2), "%")


# =========================================================
# 10. CONFUSION MATRIX
# =========================================================

print("\n====================== CONFUSION MATRIX ======================")

classes = model.classes_

print("Classes:", classes)

print(
    confusion_matrix(
        y_test,
        predictions,
        labels=classes
    )
)


# =========================================================
# 11. ACTUAL VS PREDICTED
# =========================================================

result = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})

print("\n================ ACTUAL VS PREDICTED =================")

print(result)


# =========================================================
# 12. TEST WITH A NEW ARTICLE
# =========================================================

new_title = "New artificial intelligence software launched"

new_description = (
    "Technology company introduced a powerful AI product "
    "to improve business productivity"
)


new_text = [
    new_title + " " + new_description
]


# Convert new text using the SAME vectorizer
new_text_vector = vectorizer.transform(new_text)


# Predict category
new_prediction = model.predict(new_text_vector)


print("\n====================== NEW ARTICLE TEST ======================")

print("Title       :", new_title)

print("Description :", new_description)

print("Prediction  :", new_prediction[0])