
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

base_url = (
    "https://raw.githubusercontent.com/"
    "PolyAI-LDN/task-specific-datasets/master/banking_data/"
)

df = pd.read_csv(base_url + "train.csv")

X = df["text"]
y = df["category"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=30000)),
    ("clf", LogisticRegression(max_iter=1000, random_state=42)),
])

model.fit(X_train, y_train)
predictions = model.predict(X_val)

# 1. Identify categories where performance is weakest
report = classification_report(y_val, predictions, output_dict=True, zero_division=0)

metrics = pd.DataFrame(report).T
class_metrics = metrics.loc[
    metrics.index.isin(y.unique())
]

print("\n10 categories with the lowest F1:")
print(
    class_metrics
    .sort_values("f1-score")
    [["precision", "recall", "f1-score", "support"]]
    .head(10)
    .round(3)
    .to_string()
)

# 2. Inspect incorrect predictions
errors = pd.DataFrame({
    "text": X_val.to_numpy(),
    "actual": y_val.to_numpy(),
    "predicted": predictions,
})

errors = errors[errors["actual"] != errors["predicted"]]

print(f"\nIncorrect predictions: {len(errors)}")
print("\nFirst 15 errors:")
print(errors.head(15).to_string(index=False))

errors.to_csv("validation_errors.csv", index=False)
print("\nSaved errors to validation_errors.csv")

# Count the most common actual -> predicted confusions
confusion_pairs = (
    errors.groupby(["actual", "predicted"])
    .size()
    .sort_values(ascending=False)
    .rename("count")
    .reset_index()
)

print("\nTop 15 confusion pairs:")
print(confusion_pairs.head(15).to_string(index=False))