
import time
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

base_url = (
    "https://raw.githubusercontent.com/"
    "PolyAI-LDN/task-specific-datasets/master/banking_data/"
)

df = pd.read_csv(base_url + "train.csv")

X = df["text"]
y = df["category"]

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

experiment_results = {
    "Unigram": (1,1),
    "Unigram + Bigram": (1,2),
}

results = []

for name, ngram_range in experiment_results.items():
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=ngram_range, max_features=30000)),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start

    predictions = model.predict(X_val)

    results.append({
        "Features": name,
        "Accuracy": accuracy_score(y_val, predictions),
        "Macro-F1": f1_score(
            y_val, predictions, average="macro"
        ),
        "Training seconds": train_seconds,
    })

print(pd.DataFrame(results).round(4).to_string(index=False))