import streamlit as st
import pickle
import numpy as np

# Load all models using pickle
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

# Encode categorical variables
location_v3_encoded = location_v3_options.index(location_v3)
ladder_encoded = ladder_options.index(ladder)
nart_encoded = nart_options.index(nart)

# Prepare input data for each model
input_data = {
    "minor": [ladder_encoded, bmi, location_v3_encoded, tt, tw, tl, nart_encoded],
    "major": [ladder_encoded, location_v3_encoded, tl, bmi, nart_encoded, tw],
    "ssi": [ladder_encoded, location_v3_encoded, tl, nart_encoded, lst, bmi, lsl],
    "id": [lsl, ladder_encoded, location_v3_encoded, nart_encoded, tl],
    "seroma": [ladder_encoded, location_v3_encoded, bmi, tl, tw]
}

# Function to make predictions
def make_predictions(input_data):
    predictions = {}
    for model_name, model in models.items():
        # Ensure that the input data for each model is a 2D array with the correct shape
        data = np.array(input_data[model_name]).reshape(1, -1)  # Reshaping to (1, n_features)
        
        # Log the input shape for debugging purposes
        print(f"Input shape for model {model_name}: {data.shape}")
        
        try:
            prob = model.predict_proba(data)[0][1]  # Assuming binary classification
            predictions[model_name] = round(prob * 100, 2)
        except Exception as e:
            # In case of errors, log the model name and the exception
            print(f"Error with model {model_name}: {e}")
            predictions[model_name] = "Error"
    
    return predictions


# Get predictions
if st.button("Predict"):
    predictions = make_predictions(input_data)

    # Display results with thermometer-like progress bars
    st.subheader("Predicted Probabilities")
    for outcome, prob in predictions.items():
        st.write(f"{outcome.capitalize()} Risk: {prob}%")
        st.progress(prob / 100)
