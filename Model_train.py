import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from textblob import TextBlob

# Load dataset
data_url = "https://archive.ics.uci.edu/static/public/461/data.csv"
data = pd.read_csv(data_url)

# Data Preprocessing
efficacy_mapping = {'Highly Effective': 5, 'Considerably Effective': 4, 'Moderately Effective': 3, 'Marginally Effective': 2, 'Not Effective': 1}
data['efficacy_score'] = data['effectiveness'].map(efficacy_mapping)

side_effects_mapping = {'Extremely Severe Side Effects': 5, 'Severe Side Effects': 4, 'Moderate Side Effects': 3, 'Mild Side Effects': 2, 'No Side Effects': 1}
data['side_effects_score'] = data['sideEffects'].map(side_effects_mapping)

def calculate_sentiment_safe(text):
    if isinstance(text, str):
        return TextBlob(text).sentiment.polarity
    return 0  # Default sentiment score for missing values

data['sentiment_score'] = data['commentsReview'].apply(calculate_sentiment_safe)

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

# Build Deep Learning Models
def build_model(input_size):
    input_layer = Input(shape=(1,))
    embedding = Embedding(input_size, 10)(input_layer)
    flatten = Flatten()(embedding)
    dense1 = Dense(64, activation="relu")(flatten)
    dense2 = Dense(32, activation="relu")(dense1)
    output = Dense(1, activation="linear")(dense2)
    return input_layer, output

# Condition-Based Model
condition_input, condition_output = build_model(num_conditions)
model_condition = Model(condition_input, condition_output)
model_condition.compile(optimizer="adam", loss="mean_squared_error")
model_condition.fit(data['condition_encoded'], data['rating'], epochs=10, batch_size=32)
model_condition.save("condition_model.h5")

# Drug-Based Model
drug_input, drug_output = build_model(num_drugs)
model_drug = Model(drug_input, drug_output)
model_drug.compile(optimizer="adam", loss="mean_squared_error")
model_drug.fit(data['drug_encoded'], data['rating'], epochs=10, batch_size=32)
model_drug.save("drug_model.h5")

# Hybrid Model (User + Drug)
user_input = Input(shape=(1,))
drug_input = Input(shape=(1,))
merged = Concatenate()([Embedding(num_users, 10)(user_input), Embedding(num_drugs, 10)(drug_input)])
flatten = Flatten()(merged)
dense1 = Dense(64, activation="relu")(flatten)
dense2 = Dense(32, activation="relu")(dense1)
output = Dense(1, activation="linear")(dense2)

model_hybrid = Model([user_input, drug_input], output)
model_hybrid.compile(optimizer="adam", loss="mean_squared_error")
model_hybrid.fit([data['user_id_encoded'], data['drug_encoded']], data['rating'], epochs=10, batch_size=32)
model_hybrid.save("hybrid_model.h5")

print("✅ Models trained and saved successfully!")
