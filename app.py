import requests
import streamlit as st


API_URL = "http://localhost:8000/predict"

st.set_page_config(
    page_title="AI Support Intelligence",
    page_icon="🤖",
    layout="centered",
)

st.title("AI Support Intelligence")
st.write(
    "Classify a customer support message into one of "
    "77 banking support intents."
)


ticket_text = st.text_area(
    "Customer message",
    placeholder="Example: I am still waiting on my card",
    height=150,
)

if st.button("Predict intent", type="primary"):
    if not ticket_text.strip():
        st.warning("Please enter a customer message.")
    else:
        try:
            with st.spinner("Analyzing ticket..."):
                response = requests.post(
                    API_URL,
                    json={"text": ticket_text},
                    timeout=30,
                )

            if response.status_code == 200:
                result = response.json()

                st.subheader("Prediction")
                st.success(
                    f"Predicted intent: {result['predicted_intent']}"
                )

                confidence = result["confidence"]
                st.metric(
                    "Model confidence",
                    f"{confidence:.1%}",
                )
                st.progress(confidence)

                st.caption(
                    "Confidence is the model's estimated probability, "
                    "not a guarantee that the prediction is correct."
                )

                st.subheader("Top 3 possible intents")

                for item in result["top_predictions"]:
                    st.write(
                        f"**{item['intent']}** — "
                        f"{item['confidence']:.1%}"
                    )

            else:
                st.error(
                    f"API returned HTTP {response.status_code}: "
                    f"{response.text}"
                )

        except requests.RequestException:
            st.error(
                "Could not connect to the prediction API. "
                "Make sure FastAPI is running on port 8000."
            )
