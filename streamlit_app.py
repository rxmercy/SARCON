import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load models
models = {
    "minor": pickle.load(open("minor.pkcls", "rb")),
    "major": pickle.load(open("major.pkcls", "rb")),
    "ssi": pickle.load(open("ssi.pkcls", "rb")),
    "id": pickle.load(open("id.pkcls", "rb")),
    "seroma": pickle.load(open("seroma.pkcls", "rb"))
}

# Define input options for categorical variables
location_v3_options = [
    "shoulder", "upperarm or elbow", "forearm or wrist",
    "hand", "thigh or knee", "lower leg or ankle", "foot or toe"
]
ladder_options = [
    "primary closure", "complex repair", "sg",
    "local flap", "pedicled flap", "free flap"
]
nart_options = ["no", "yes"]

# Function to one-hot encode categorical variables
def one_hot_encode(value, options):
    encoding = [0] * len(options)
    encoding[options.index(value)] = 1
    return encoding

# Streamlit UI
st.title("Predictive App for Surgical Outcomes")

# Input form
location_v3 = st.selectbox("STS Location", location_v3_options)
ladder = st.select_slider("Reconstructive Ladder", options=ladder_options)
bmi = st.number_input("Body Mass Index", min_value=10.0, max_value=50.0, step=0.1)
tl = st.number_input("Tumor Length (craniocaudal), cm", min_value=0.0)
tw = st.number_input("Tumor Width (mediolateral), cm", min_value=0.0)
tt = st.number_input("Tumor Thickness (anteroposterior), cm", min_value=0.0)
nart = st.selectbox("Neoadjuvant Radiotherapy", nart_options)
lst = st.number_input("Limb Segment Length (craniocaudal), cm", min_value=0.0)
lsl = st.number_input("Limb Segment Thickness (anteroposterior), cm", min_value=0.0)

# One-hot encode categorical variables
location_v3_encoded = one_hot_encode(location_v3, location_v3_options)
ladder_encoded = one_hot_encode(ladder, ladder_options)
nart_encoded = one_hot_encode(nart, nart_options)

# Prepare input data for each model
input_data = {
    "minor": ladder_encoded + location_v3_encoded + [bmi, tt, tw, tl] + nart_encoded,
    "major": ladder_encoded + location_v3_encoded + [tl, bmi] + nart_encoded + [tw],
    "ssi": ladder_encoded + location_v3_encoded + [tl] + nart_encoded + [lst, bmi, lsl],
    "id": [lsl] + ladder_encoded + location_v3_encoded + nart_encoded + [tl],
    "seroma": ladder_encoded + location_v3_encoded + [bmi, tl, tw]
}

# Function to make predictions
def make_predictions(input_data):
    predictions = {}
    for model_name, model in models.items():
        data = np.array(input_data[model_name]).reshape(1, -1)  # Reshape input to 2D array
        try:
            prob = model.predict_proba(data)[0][1]  # Probability of class 1 (positive outcome)
            predictions[model_name] = round(prob * 100, 2)
        except Exception as e:
            st.error(f"Error with {model_name} model: {e}")
            predictions[model_name] = "Error"
    return predictions

# Display predictions
if st.button("Predict"):
    predictions = make_predictions(input_data)

    st.subheader("Predicted Probabilities")
    for outcome, prob in predictions.items():
        if prob == "Error":
            st.write(f"Error with {outcome.capitalize()} prediction.")
        else:
            st.write(f"{outcome.capitalize()} Risk: {prob}%")
            if isinstance(prob, (int, float)) and 0 <= prob <= 100:
                st.progress(prob / 100)
            else:
                st.write("Invalid probability value. Please check model outputs.")
