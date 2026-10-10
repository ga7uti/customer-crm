import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# 1. Load the official training dataset
base_url = (
    "https://raw.githubusercontent.com/"
    "PolyAI-LDN/task-specific-datasets/master/banking_data/"
)

train_df = pd.read_csv(base_url + "train.csv")

X_train = train_df["text"]
y_train = train_df["category"]

print(f"Training examples: {len(train_df)}")
print(f"Intent categories: {y_train.nunique()}")


# 2. Build the complete ML pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 1),
            max_features=30000
        )
    ),
    (
        "classifier",
        LogisticRegression(
            C=10.0,
            max_iter=1000,
            random_state=42
        )
    )
])


# 3. Train the pipeline
model.fit(X_train, y_train)

print("Model training completed.")


# 4. Save the trained model to disk
model_path = "ticket_intent_model.joblib"
joblib.dump(model, model_path)
print(f"Trained model saved to: {model_path}")