import joblib

#Load the trained model from disk
model = joblib.load("ticket_intent_model.joblib")

def predict_ticket(text):
    if not text or not text.strip():
        raise ValueError("Ticket text cannot be empty.")

    #predict the most likely intent
    predicted_intent = model.predict([text])[0]

    #get the probability of the predicted intent
    probabilities = model.predict_proba([text])[0]
    confidence = float(max(probabilities))

    return {
        "ticket_text": text,
        "predicted_intent": predicted_intent,
        "confidence": confidence
    }

if __name__ == "__main__":
    ticket = input("Enter a customer support message: ")

    result = predict_ticket(ticket)

    print("\nPrediction")
    print(f"Intent: {result['predicted_intent']}")
    print(f"Confidence: {result['confidence']:.2%}")