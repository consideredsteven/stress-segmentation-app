import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("cluster_classifier.pkl")

clf = load_model()

# Load feature names (assuming the model was trained on your dataset)
feature_names = ['prop_01', 'prop_02', 'prop_03', 'prop_04', 'prop_05',
                 'cope_06', 'cope_prob_07', 'cope_prob_08', 'cope_emot_09',
                 'cope_emot_10', 'cope_emot_11', 'cope_avoid_12', 'cope_avoid_13',
                 'cope_avoid_14', 'perf_15', 'perf_16', 'perf_17', 'perf_18',
                 'perf_19', 'hours_20']

# UI Layout
st.title("Workplace Stress Segment Classifier")
st.write("Enter responses to predict the respondent’s stress segment.")

# User input fields
user_input = []
for feature in feature_names:
    user_input.append(st.slider(f"{feature}", min_value=1, max_value=5, value=3))

# Predict segment
if st.button("Predict Segment"):
    new_data = pd.DataFrame([user_input], columns=feature_names)
    predicted_cluster = clf.predict(new_data)[0]
    st.success(f"Predicted Segment: **Cluster {predicted_cluster}**")

