import streamlit as st
import pickle
import numpy as np

# Load models
models = {
    "minor": pickle.load(open("models/minor.pkcls", "rb")),
    "major": pickle.load(open("models/major.pkcls", "rb")),
    "ssi": pickle.load(open("models/ssi.pkcls", "rb")),
    "id": pickle.load(open("models/id.pkcls", "rb")),
    "seroma": pickle.load(open("models/seroma.pkcls", "rb"))
}

# Input options for categorical variables
location_v3_options = [
    "shoulder", "upperarm or elbow", "forearm or wrist",
    "hand", "thigh or knee", "lower leg or ankle", "foot or toe"
]
ladder_options = [
    "primary closure", "complex repair", "sg",
    "local flap", "pedicled flap", "free flap"
]
nart_options = ["no", "yes"]

# Define input fields
st.title("Surgical Risk Prediction App")

# User input
location_v3 = st.selectbox("STS Location", location_v3_options)
ladder = st.select_slider("Reconstructive Ladder", options=ladder_options)
bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, step=0.1)
tl = st.number_input("Tumor Length (cm)", min_value=0.0)
tw = st.number_input("Tumor Width (cm)", min_value=0.0)
tt = st.number_input("Tumor Thickness (cm)", min_value=0.0)
nart = st.selectbox("Neoadjuvant Radiotherapy", nart_options)
lst = st.number_input("Limb Segment Length (cm)", min_value=0.0)
lsl = st.number_input("Limb Segment Thickness (cm)", min_value=0.0)

# Encode categorical variables
location_v3_encoded = location_v3_options.index(location_v3)
ladder_encoded = ladder_options.index(ladder)
nart_encoded = nart_options.index(nart)

# Prepare input data for each model
input_data = {
    "minor": [ladder_encoded, location_v3_encoded, bmi, tt, tw, tl, nart_encoded],
    "major": [ladder_encoded, location_v3_encoded, tl, bmi, nart_encoded, tw],
    "ssi": [ladder_encoded, location_v3_encoded, tl, nart_encoded, lst, bmi, lsl],
    "id": [lsl, ladder_encoded, location_v3_encoded, nart_encoded, tl],
    "seroma": [ladder_encoded, location_v3_encoded, bmi, tl, tw, nart_encoded]
}

# Prediction function
def make_predictions(input_data):
    predictions = {}
    for model_name, model in models.items():
        data = np.array(input_data[model_name]).reshape(1, -1)
        try:
            prob = model.predict_proba(data)[0][1] * 100  # Probability of positive outcome
            predictions[model_name] = round(prob, 2)
        except Exception as e:
            st.error(f"Error with {model_name} model: {e}")
            predictions[model_name] = None
    return predictions

# Generate predictions
if st.button("Predict"):
    predictions = make_predictions(input_data)

    # Display predictions
    st.subheader("Predicted Probabilities")
    for outcome, prob in predictions.items():
        if prob is None:
            st.write(f"{outcome.capitalize()} prediction error.")
        else:
            st.write(f"{outcome.capitalize()} Risk: {prob}%")
            st.progress(int(prob))

