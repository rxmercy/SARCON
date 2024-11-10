import streamlit as st
import pickle
import numpy as np

# Load models
with open('minor.pkcls', 'rb') as file:
    minor_model = pickle.load(file)
with open('major.pkcls', 'rb') as file:
    major_model = pickle.load(file)
with open('ssi.pkcls', 'rb') as file:
    ssi_model = pickle.load(file)
with open('id.pkcls', 'rb') as file:
    id_model = pickle.load(file)
with open('seroma.pkcls', 'rb') as file:
    seroma_model = pickle.load(file)

# User Input
st.title("Prediction Models for Independent Outcomes")

location_v3 = st.selectbox("STS Location", ["shoulder, upperarm or elbow", "forearm or wrist", "hand", "thigh or knee", "lower leg or ankle", "foot or toe"])
bmi = st.number_input("Body mass index")
nart = st.selectbox("Neoadjuvant radiotherapy", ["no", "yes"])
tl = st.number_input("Tumor length (craniocaudal), cm")
tw = st.number_input("Tumor width (mediolateral), cm")
tt = st.number_input("Tumor thickness (anteroposterior), cm")
lst = st.number_input("Limb segment length (craniocaudal), cm")
lsl = st.number_input("Limb segment thickness (anteroposterior), cm")
ladder = st.slider("Reconstructive ladder", min_value=1, max_value=6)  # Assuming slider values from 1 to 6

# Make predictions
minor_input = [ladder, bmi, location_v3, tt, tw, tl, nart]
major_input = [ladder, location_v3, tl, bmi, nart, tw]
ssi_input = [ladder, location_v3, tl, nart, lst, bmi, lsl]
id_input = [lsl, ladder, location_v3, nart, tl]
seroma_input = [ladder, location_v3, bmi, tl, tw]

minor_pred = minor_model.predict_proba(np.array(minor_input).reshape(1, -1))[0, 1]
major_pred = major_model.predict_proba(np.array(major_input).reshape(1, -1))[0, 1]
ssi_pred = ssi_model.predict_proba(np.array(ssi_input).reshape(1, -1))[0, 1]
id_pred = id_model.predict_proba(np.array(id_input).reshape(1, -1))[0, 1]
seroma_pred = seroma_model.predict_proba(np.array(seroma_input).reshape(1, -1))[0, 1]

# Display results
st.write("## Predictions")
st.write("### Minor Outcome")
st.progress(minor_pred)

st.write("### Major Outcome")
st.progress(major_pred)

st.write("### SSI Outcome")
st.progress(ssi_pred)

st.write("### ID Outcome")
st.progress(id_pred)

st.write("### Seroma Outcome")
st.progress(seroma_pred)
