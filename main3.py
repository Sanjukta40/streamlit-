import streamlit as st
import pandas as pd
import pickle

# Function to load the pickle file
def load_pickle_file(uploaded_file):
    return pickle.load(uploaded_file)

# Streamlit app layout
st.title("Load and Visualize Data from Pickle Files")

# File upload section
st.sidebar.title("Upload Pickle Files")
train_file = st.sidebar.file_uploader("Upload training dataset (PM_train.pkl)", type=["pkl"])
train_updated_file = st.sidebar.file_uploader("Upload updated training dataset (PM_train_updated.pkl)", type=["pkl"])
test_file = st.sidebar.file_uploader("Upload test dataset (PM_test.pkl)", type=["pkl"])
truth_file = st.sidebar.file_uploader("Upload truth dataset (Truth_Value.pkl)", type=["pkl"])


