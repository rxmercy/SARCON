import streamlit as st
import pickle
import numpy as np

# Load the minor complications model
with open('minor.pkcls', 'rb') as file:
    minor_model = pickle.load(file)

# User Input
st.title("Prediction of Minor Complications")

location_v3 = st.selectbox("STS Location", ["shoulder, upperarm or elbow", "forearm or wrist", "hand", "thigh or knee", "lower leg or ankle", "foot or toe"])
bmi = st.number_input("Body mass index")
nart = st.selectbox("Neoadjuvant radiotherapy", ["no", "yes"])
tl = st.number_input("Tumor length (craniocaudal), cm")
tw = st.number_input("Tumor width (mediolateral), cm")
tt = st.number_input("Tumor thickness (anteroposterior), cm")
ladder = st.select_slider("Reconstructive ladder", options=["primary closure", "complex repair", "sg", "local flap", "pedicled flap", "free flap"])

# Encode categorical variables as they were during model training
location_v3_mapping = {
    "shoulder, upperarm or elbow": 0,
    "forearm or wrist": 1,
    "hand": 2,
    "thigh or knee": 3,
    "lower leg or ankle": 4,
    "foot or toe": 5
}
nart_mapping = {
    "no": 0,
    "yes": 1
}
ladder_mapping = {
    "primary closure": 0,
    "complex repair": 1,
    "sg": 2,
    "local flap": 3,
    "pedicled flap": 4,
    "free flap": 5
}

# Convert inputs to numeric values
location_v3_num = location_v3_mapping[location_v3]
nart_num = nart_mapping[nart]
ladder_num = ladder_mapping[ladder]

# Prepare model inputs
minor_input = [ladder_num, location_v3_num, bmi, tt, tw, tl, nart_num]

# Make prediction
minor_pred = minor_model.predict_proba(np.array(minor_input).reshape(1, -1))[0, 1]

# Display result
st.write("## Prediction of Minor Complications")
st.write("### Probability of Minor Complications")
st.progress(minor_pred)
