import streamlit as st
import pickle
import numpy as np
import Orange

# Load all models
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

# Create the input form
st.title("Predictive App for Surgical Outcomes")

# Input fields
location_v3 = st.selectbox("STS Location", location_v3_options)
ladder = st.select_slider("Reconstructive Ladder", options=ladder_options)
bmi = st.number_input("Body Mass Index", min_value=10.0, max_value=50.0, step=0.1)
tl = st.number_input("Tumor Length (craniocaudal), cm", min_value=0.0)
tw = st.number_input("Tumor Width (mediolateral), cm", min_value=0.0)
tt = st.number_input("Tumor Thickness (anteroposterior), cm", min_value=0.0)
nart = st.selectbox("Neoadjuvant Radiotherapy", nart_options)
lst = st.number_input("Limb Segment Length (craniocaudal), cm", min_value=0.0)
lsl = st.number_input("Limb Segment Thickness (anteroposterior), cm", min_value=0.0)

# Encode categorical variables as integer indices
location_v3_encoded = location_v3_options.index(location_v3)
ladder_encoded = ladder_options.index(ladder)
nart_encoded = nart_options.index(nart)

# Prepare input data for each model
input_data = {
    "minor": [ladder_encoded, location_v3_encoded, bmi, tt, tw, tl, nart_encoded],
    "major": [ladder_encoded, location_v3_encoded, tl, bmi, nart_encoded, tw],
    "ssi": [ladder_encoded, location_v3_encoded, tl, nart_encoded, lst, bmi, lsl],
    "id": [lsl, ladder_encoded, location_v3_encoded, nart_encoded, tl],
    "seroma": [ladder_encoded, location_v3_encoded, bmi, tl, tw]
}

# Function to make predictions
def make_predictions(input_data):
    predictions = {}
    for model_name, model in models.items():
        data = np.array(input_data[model_name]).reshape(1, -1)  # Reshape input to 2D array (1 sample)

        # Convert input data to Orange Table format (same as the example)
        if model_name == "minor":
            domain = Orange.data.Domain([
                Orange.data.DiscreteVariable("ladder", ladder_options),
                Orange.data.DiscreteVariable("location_v3", location_v3_options),
                Orange.data.ContinuousVariable("bmi"),
                Orange.data.ContinuousVariable("tt"),
                Orange.data.ContinuousVariable("tw"),
                Orange.data.ContinuousVariable("tl"),
                Orange.data.DiscreteVariable("nart", nart_options)
            ])
        elif model_name == "major":
            domain = Orange.data.Domain([
                Orange.data.DiscreteVariable("ladder", ladder_options),
                Orange.data.DiscreteVariable("location_v3", location_v3_options),
                Orange.data.ContinuousVariable("tl"),
                Orange.data.ContinuousVariable("bmi"),
                Orange.data.DiscreteVariable("nart", nart_options),
                Orange.data.ContinuousVariable("tw")
            ])
        # Add more conditions for other models as necessary

        # Create Orange Table from input data
        orange_data = Orange.data.Table(domain, [data])

        # Make predictions using the model
        try:
            prob = model.predict_proba(orange_data)[0][1]  # Assuming binary classification
            predictions[model_name] = round(prob * 100, 2)
        except Exception as e:
            print(f"Error with model {model_name}: {e}")
            predictions[model_name] = f"Error: {e}"
    return predictions

# Get predictions
if st.button("Predict"):
    predictions = make_predictions(input_data)

    # Display results with thermometer-like progress bars
    st.subheader("Predicted Probabilities")
    for outcome, prob in predictions.items():
        if "Error" in str(prob):
            st.write(f"Error with model {outcome.capitalize()} prediction: {prob}")
        else:
            st.write(f"{outcome.capitalize()} Risk: {prob}%")
            if isinstance(prob, (int, float)) and 0 <= prob <= 100:
                st.progress(prob / 100)  # Display progress bar for valid probability values
            else:
                st.write("Invalid probability value. Please check model outputs.")
