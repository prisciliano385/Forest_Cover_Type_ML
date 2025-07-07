import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler

# Load the classification machine learning model used for prediction:
with open("../notebooks/models/ada_boost_best_model.pkl", "rb") as file:
    ada = pickle.load(file)
    
with open("../notebooks/models/standard_scaler.pkl", "rb") as file:
    st_scaler = pickle.load(file)

st.title("Forest Cover Type Prediction")            
            
st.markdown("""
This app, the final stage of this Machine Learning project, is designed
to receive input variables from the user and predict the forest cover
type of some piece of land located in the Roosevelt National Forest.

10 numerical values and 2 categorical variables must be provided by 
the user in order for the model to predict the actual forest 
cover type.
""")

with st.form("Parameters:"):
    st.subheader("Numerical Inputs")
    elevation = st.number_input("Elevation (meters)", min_value=0)
    aspect = st.number_input("Aspect (azimuth in degrees)", min_value=0, max_value=360)
    slope = st.number_input("Slope (degrees)", min_value=0)
    horizontal_distance_to_hydrology = st.number_input("Horizontal Distance to Hydrology (meters)", min_value=0)
    vertical_distance_to_hydrology = st.number_input("Vertical Distance to Hydrology (meters)")
    horizontal_distance_to_roadways = st.number_input("Horizontal Distance to Roadways (meters)", min_value=0)
    hillshade_9am = st.number_input("Hillshade at 9 AM", min_value=0, max_value=255)
    hillshade_noon = st.number_input("Hillshade at Noon", min_value=0, max_value=255)
    hillshade_3pm = st.number_input("Hillshade at 3 PM", min_value=0, max_value=255)
    horizontal_distance_to_fire_points = st.number_input("Horizontal Distance to Fire Points (meters)", min_value=0)

    st.subheader("Categorical Inputs")
    wilderness_area = st.selectbox("Wilderness Area", ["Rawah", "Neota", "Comanche Peak", "Cache la Poudre"])
    soil_type = st.selectbox("Soil Type", [f"Soil Type {i}" for i in range(1,41)])

    submitted = st.form_submit_button("Predict Forest Cover Type")
    # This returns a boolean when pressed

if submitted:
    # Manually one-hot encode Wilderness Area
    areas = ["Rawah", "Neota", "Comanche Peak", "Cache la Poudre"]
    wild_list = [0] * 4
    index = areas.index(wilderness_area)
    wild_list[index] = 1
    
    # Manually one-hot encode Soil Type
    soil_number = int(soil_type.split()[-1])
    soil_list = np.zeros(40)
    soil_list[soil_number-1] = 1
    
    # Combine all data:
    numerical_inputs = [
            elevation, aspect, slope,
            horizontal_distance_to_hydrology, vertical_distance_to_hydrology,
            horizontal_distance_to_roadways,
            hillshade_9am, hillshade_noon, hillshade_3pm,
            horizontal_distance_to_fire_points
        ]
    # Save the data to a numpy 1D array
    data = np.array(numerical_inputs + wild_list + list(soil_list)).reshape(1, -1)
    # ADI: zergatik egiten dugu reshape? st_scaler-rek 2D-ko array jasotzen duelako
    
    # Scale the data
    data = st_scaler.transform(data)
    
    # Predict the Tree Cover Type
    prediction = ada.predict(data)[0] # predict devuelve un array
    
cover_types = {
            1: "Spruce/Fir",
            2: "Lodgepole Pine",
            3: "Ponderosa Pine",
            4: "Cottonwood/Willow",
            5: "Aspen",
            6: "Douglas-fir",
            7: "Krummholz"
        }

st.success(f"Predicted Forest Cover Type: **{cover_types.get(prediction, 'Unknown')}**")