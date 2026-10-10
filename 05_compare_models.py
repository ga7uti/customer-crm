import time
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

base_url = (
    "https://raw.githubusercontent.com/" 
    "PolyAI-LDN/task-specific-datasets/master/banking_data/"
)

df = pd.read_csv(base_url + "train.csv")

X = df["text"]
y = df["category"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


models = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "decision_tree": DecisionTreeClassifier(max_depth=30, min_samples_leaf=2, random_state=42),
}

result = []

for name, classifier in models.items():
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=30000)),
        ("clf", classifier),
    ])

    start_time = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start_time

    predictions = pipeline.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    f1 = f1_score(y_val, predictions, average='weighted')

    result.append({
        "model": name,
        "train_time": train_seconds,
        "accuracy": accuracy,
        "f1_score": f1
    })

    print(f"\n{name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {f1:.4f}")
    print(f"Training time: {train_seconds:.2f} seconds")

print("\n=== Model comparison ===")
print(
    pd.DataFrame(result)
    .sort_values("f1_score", ascending=False)
    .round(4)
    .to_string(index=False)
)
