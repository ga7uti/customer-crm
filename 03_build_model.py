import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score

base_url = (
    "https://raw.githubusercontent.com/" 
    "PolyAI-LDN/task-specific-datasets/master/banking_data/"
)

train_df = pd.read_csv(base_url + "train.csv")
test_df = pd.read_csv(base_url + "test.csv")


X = train_df["text"]
y = train_df["category"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=30000)),
    ("clf", LogisticRegression(max_iter=1000, random_state=42)),
])

model.fit(X_train, y_train)

predictions = model.predict(X_val)

accuracy = accuracy_score(y_val, predictions)
macro_f1 = f1_score(y_val, predictions, average="macro")

print(f"Training examples: {len(X_train)}")
print(f"Validation examples: {len(X_val)}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro-F1: {macro_f1:.4f}")

print("\nSample predictions:")
for text, actual, predicted in zip(
    X_val.iloc[:10],
    y_val.iloc[:10],
    predictions[:10]
):
    print(f"\nText: {text}")
    print(f"Actual: {actual}")
    print(f"Predicted: {predicted}")