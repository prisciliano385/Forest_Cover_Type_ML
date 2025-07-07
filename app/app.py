import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.title("Forest Cover Type Prediction")

st.markdown("""
            This app, the final stage of this Machine Learning project, is designed
            to receive input variables from the user and predict the forest cover
            type of some piece of land located in the Roosevelt National Forest.
            
            It is important to note that the predictions are only valid for the
            Roosvelt National Forest, as the model used for predictions was trained
            only with data belonging to the aforementioned forest.
            
            10 numerical values and 2 categories must be provided by the user
            in order for the model to predict the actual forest cover type.""")