import streamlit as st
import requests

# Backend URL (ensure FastAPI is running)
BACKEND_URL = "http://127.0.0.1:8080"

st.title("🛡️ Hate Speech Classifier")

# Train the model
st.subheader("📌 Train the Model")
if st.button("🚀 Start Training"):
    response = requests.get(f"{BACKEND_URL}/train")
    if response.status_code == 200:
        st.success("Training successful!")
    else:
        st.error(f"Error: {response.text}")

# Prediction Section
st.subheader("🔍 Predict Hate Speech")
user_input = st.text_area("✍️ Enter text to classify:")

if st.button("🔮 Predict"):
    with st.spinner("Analyzing text... 🧠"):
        if user_input:
            response = requests.post(f"{BACKEND_URL}/predict", params={"words": user_input})
            if response.status_code == 200:
                prediction = response.json().get("prediction", "No prediction")
                st.write(f"**Prediction:** {prediction}")
            else:
                st.error(f"Error: {response.text}")
        else:
            st.warning("Please enter some text.")

st.write("Backend running at:", BACKEND_URL)
