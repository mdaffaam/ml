import pandas as pd
import numpy as np
import json
import joblib
from flask import Flask, jsonify, request

# Importing TensorFlow and Keras components
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Initialize Flask application
app = Flask(__name__)

# Load the trained model and LabelEncoder
model = load_model('food_recommendation_model.h5')
label_encoder = joblib.load('label_encoder.pkl')

# Define a function for preprocessing input data
def preprocess_input(data):
    # Assuming data is a dictionary containing input values
    processed_data = {
        'Is_Diet': int(data['Is_Diet']),
        'Is_Bulking': int(data['Is_Bulking']),
        'Calories (kcal)': float(data['Calories']),
        'Protein (g)': float(data['Protein']),
        'Carbohydrates (g)': float(data['Carbohydrates']),
        'Fats (g)': float(data['Fats'])
    }
    return processed_data

# Define a function for making predictions
def predict(data):
    processed_data = preprocess_input(data)
    input_features = np.array([list(processed_data.values())])
    prediction = model.predict(input_features)
    predicted_category = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    return predicted_category

# Define a route for receiving input and returning predictions
@app.route('/predict', methods=['POST'])
def make_prediction():
    data = request.get_json()
    prediction = predict(data)
    return jsonify({'prediction': prediction})

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
