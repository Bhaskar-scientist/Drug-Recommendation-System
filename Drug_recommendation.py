import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from textblob import TextBlob

# Load preprocessed dataset
data_url = "https://archive.ics.uci.edu/static/public/461/data.csv"
data = pd.read_csv(data_url)

# Preprocessing
# Mapping effectiveness and side effects to numerical scores
efficacy_mapping = {
    'Highly Effective': 5, 
    'Considerably Effective': 4, 
    'Moderately Effective': 3, 
    'Marginally Effective': 2, 
    'Not Effective': 1
}
data['efficacy_score'] = data['effectiveness'].map(efficacy_mapping)

side_effects_mapping = {
    'Extremely Severe Side Effects': 5,
    'Severe Side Effects': 4,
    'Moderate Side Effects': 3,
    'Mild Side Effects': 2,
    'No Side Effects': 1
}
data['side_effects_score'] = data['sideEffects'].map(side_effects_mapping)

def calculate_sentiment_safe(text):
    if isinstance(text, str):
        return TextBlob(text).sentiment.polarity
    return 0  # Default sentiment score for missing values

data['sentiment_score'] = data['commentsReview'].apply(calculate_sentiment_safe)

# Drop missing values
data.dropna(subset=['efficacy_score', 'sentiment_score', 'side_effects_score', 'rating'], inplace=True)

# Normalize numerical features
scaler = MinMaxScaler()
data[['efficacy_score', 'sentiment_score', 'side_effects_score']] = scaler.fit_transform(
    data[['efficacy_score', 'sentiment_score', 'side_effects_score']]
)

# Encode categorical data
user_encoder = LabelEncoder()
drug_encoder = LabelEncoder()
condition_encoder = LabelEncoder()

data['user_id_encoded'] = user_encoder.fit_transform(data['reviewID'])
data['drug_encoded'] = drug_encoder.fit_transform(data['urlDrugName'])
data['condition_encoded'] = condition_encoder.fit_transform(data['condition'])

num_users = len(user_encoder.classes_)
num_drugs = len(drug_encoder.classes_)
num_conditions = len(condition_encoder.classes_)

# Load trained deep learning models (assuming models are trained separately and saved)
model_condition = keras.models.load_model("condition_model.h5")
model_drug = keras.models.load_model("drug_model.h5")
model_hybrid = keras.models.load_model("hybrid_model.h5")

# Streamlit App UI
st.title("Drug Recommendation System")
st.write("Enter your condition, drug name, or both to get recommendations.")

# User inputs
condition_input = st.text_input("Enter Condition (optional)")
drug_input = st.text_input("Enter Drug Name (optional)")

# Function for Condition-Based Recommendation
def recommend_by_condition(condition, top_n=5):
    if condition not in condition_encoder.classes_:
        return "Condition not found in database."
    condition_idx = condition_encoder.transform([condition])[0]
    predictions = model_condition.predict(np.array([condition_idx]))
    top_drugs = predictions.argsort()[::-1][:top_n]
    return [drug_encoder.inverse_transform([i])[0] for i in top_drugs]

# Function for Drug-Based Recommendation
def recommend_by_drug(drug, top_n=5):
    if drug not in drug_encoder.classes_:
        return "Drug not found in database."
    drug_idx = drug_encoder.transform([drug])[0]
    predictions = model_drug.predict(np.array([drug_idx]))
    top_drugs = predictions.argsort()[::-1][:top_n]
    return [drug_encoder.inverse_transform([i])[0] for i in top_drugs]

# Function for Hybrid Recommendation
def recommend_hybrid(user_id, drug, top_n=5):
    try:
        user_idx = user_encoder.transform([user_id])[0]
        drug_idx = drug_encoder.transform([drug])[0]
    except ValueError:
        return "User or Drug not found in database."
    
    predictions = model_hybrid.predict(np.array([[user_idx, drug_idx]]))
    top_drugs = predictions.argsort()[::-1][:top_n]
    return [drug_encoder.inverse_transform([i])[0] for i in top_drugs]

# Display results
if st.button("Get Recommendations"):
    if condition_input and not drug_input:
        st.write("**Condition-Based Recommendations:**", recommend_by_condition(condition_input))
    elif drug_input and not condition_input:
        st.write("**Drug-Based Recommendations:**", recommend_by_drug(drug_input))
    elif condition_input and drug_input:
        st.write("**Hybrid Recommendations:**", recommend_hybrid("39483", drug_input))  # Default user_id for now
    else:
        st.write("Please enter at least one input.")
