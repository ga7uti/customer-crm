
import joblib
import numpy as np

MODEL_PATH = "ticket_intent_model.joblib"

# Load the existing trained pipeline.
model = joblib.load(MODEL_PATH)

# Get the two stages of the pipeline.
vectorizer = model.named_steps["tfidf"]
classifier = model.named_steps["classifier"]

text = input("Enter a customer support message: ").strip()

if not text:
    raise SystemExit("Please enter a non-empty message.")

# Convert the message into the same TF-IDF features used during training.
X = vectorizer.transform([text])

# Get the classifier's scores for all possible intents.
scores = classifier.decision_function(X)[0]
ranked_indices = np.argsort(scores)[::-1]

predicted_idx = ranked_indices[0]
runner_up_idx = ranked_indices[1]

predicted_intent = classifier.classes_[predicted_idx]
runner_up_intent = classifier.classes_[runner_up_idx]

print("\nPrediction analysis")
print("-" * 50)
print(f"Message: {text}")
print(f"Predicted intent: {predicted_intent}")
print(f"Runner-up intent: {runner_up_intent}")
print(f"Predicted class score: {scores[predicted_idx]:.4f}")
print(f"Runner-up class score: {scores[runner_up_idx]:.4f}")

# Compare the learned weights of the two competing intents.
# A positive contribution favors the predicted intent.
# A negative contribution favors the runner-up.
weight_difference = (
    classifier.coef_[predicted_idx]
    - classifier.coef_[runner_up_idx]
)

feature_names = vectorizer.get_feature_names_out()
feature_values = X.toarray()[0]

contributions = feature_values * weight_difference

positive_indices = np.argsort(contributions)[::-1]
negative_indices = np.argsort(contributions)

print(f"\nWords/features favoring {predicted_intent}:")
count = 0
for idx in positive_indices:
    if contributions[idx] <= 0:
        break
    if feature_values[idx] > 0:
        print(f"  {feature_names[idx]}: +{contributions[idx]:.4f}")
        count += 1
        if count == 10:
            break

print(f"\nWords/features favoring {runner_up_intent}:")
count = 0
for idx in negative_indices:
    if contributions[idx] >= 0:
        break
    if feature_values[idx] > 0:
        print(f"  {feature_names[idx]}: {contributions[idx]:.4f}")
        count += 1
        if count == 10:
            break
