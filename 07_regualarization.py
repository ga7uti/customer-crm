
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

results = []

for c_value in [0.1, 1.0, 10.0]:
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 1),
            max_features=30000,
        )),
        ("classifier", LogisticRegression(
            C=c_value,
            max_iter=1000,
            random_state=42,
        )),
    ])

    start = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - start

    predictions = model.predict(X_val)

    results.append({
        "C": c_value,
        "Accuracy": accuracy_score(y_val, predictions),
        "Macro-F1": f1_score(
            y_val, predictions, average="macro"
        ),
        "Training seconds": elapsed,
    })

print(pd.DataFrame(results).round(4).to_string(index=False))