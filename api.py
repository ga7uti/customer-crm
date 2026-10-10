from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel,Field, field_validator

MODEL_PATH = Path(__file__).parent / "ticket_intent_model.joblib"

model = joblib.load(MODEL_PATH)


app = FastAPI(
    title="AI Support Intelligence API",
    description="Predict the intent of a customer support message.",
    version="1.0.0",
)

class TicketRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The customer's support message",
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Ticket text cannot be blank.")
        return value.strip()



class PredictionResponse(BaseModel):
    predicted_intent: str
    confidence: float
    top_predictions: list[dict[str, float | str]]


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
def predict_ticket(request: TicketRequest):
    try:
        predicted_intent = model.predict([request.text])[0]
        probabilities = model.predict_proba([request.text])[0]
        classes = model.named_steps["classifier"].classes_

        top_indices = np.argsort(probabilities)[::-1][:3]

        top_predictions = [
            {"intent": classes[i], "confidence": float(probabilities[i])}
            for i in top_indices
        ]

        return PredictionResponse(
            predicted_intent=predicted_intent,
            confidence=float(max(probabilities)),
            top_predictions=top_predictions,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e 