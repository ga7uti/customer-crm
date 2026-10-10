
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

base_url = (
    "https://raw.githubusercontent.com/"
    "PolyAI-LDN/task-specific-datasets/master/banking_data/"
)

train_df = pd.read_csv(base_url + "train.csv")
test_df = pd.read_csv(base_url + "test.csv")

X_train = train_df["text"]
y_train = train_df["category"]

X_test = test_df["text"]
y_test = test_df["category"]

model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=30000)),
    ("clf", LogisticRegression(C=10.0, max_iter=1000, random_state=42)),
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)


print("Official test examples:", len(X_test))
print(f"Test accuracy: {accuracy_score(y_test, predictions):.4f}")
print(
    "Test macro-F1:",
    f"{f1_score(y_test, predictions, average='macro'):.4f}",
)

print("\nPer-category performance:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0,
    digits=3,
))

