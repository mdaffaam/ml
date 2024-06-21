from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
import json
import joblib

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

app = Flask(__name__)


model = load_model('food_recommendation_model.h5')
label_encoder = joblib.load('label_encoder.pkl')

# Load datasets
foods = pd.read_csv('Food.csv')
beverages = pd.read_csv('Beverage.csv')

# Function to preprocess amount column
def preprocess_amount(amount):
    if isinstance(amount, str):
        return float(amount.split('/')[0].replace('g', '').replace('mL', ''))
    else:
        return amount

# Function to filter food based on criteria
def filter_food(height, weight, desired_weight, is_diet, num_choices=50):
    # Calculate the caloric needs
    if is_diet:
        calories_needed = (weight - desired_weight) * 7700 / 30  # Simple weight loss estimation
        category = 'Diet'
    else:
        calories_needed = (desired_weight - weight) * 7700 / 30  # Simple weight gain estimation
        category = 'Bulking'

    # Filter food data
    filtered_food = foods[(foods['Category'] == category) & (foods['Calories (kcal)'] <= calories_needed)].sample(min(num_choices, len(foods)))

    # Preprocess amount columns
    filtered_food['Amount for Diet (g/mL)'] = filtered_food['Amount for Diet (g/mL)'].apply(preprocess_amount)
    filtered_food['Amount for Bulking (g/mL)'] = filtered_food['Amount for Bulking (g/mL)'].apply(preprocess_amount)

    # Create feature for diet or bulking
    filtered_food['Is_Diet'] = filtered_food['Amount for Diet (g/mL)'].notnull().astype(int)
    filtered_food['Is_Bulking'] = filtered_food['Amount for Bulking (g/mL)'].notnull().astype(int)

    # Input features for model prediction
    X_food = filtered_food[['Is_Diet', 'Is_Bulking', 'Calories (kcal)', 'Protein (g)', 'Carbohydrates (g)', 'Fats (g)']]

    # Predict using the model
    predictions = model.predict(X_food)
    predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

    # Add predicted labels to filtered_food dataframe
    filtered_food['Predicted_Category'] = predicted_labels

    # Format output as required
    formatted_food = []
    for _, row in filtered_food.iterrows():
        formatted_item = {
            "Food Name": row['Food Name'],
            "Food Type": row['Food Type'],
            "Ingredients": row['Ingredients'],
            "Spices": row['Spices'],
            "Calories (kcal)": row['Calories (kcal)'],
            "Protein (g)": row['Protein (g)'],
            "Carbohydrates (g)": row['Carbohydrates (g)'],
            "Fats (g)": row['Fats (g)'],
            "Predicted Category": row['Predicted_Category']
        }
        formatted_food.append(formatted_item)

    return formatted_food

# Function to filter beverages based on criteria
def filter_beverages(height, weight, desired_weight, is_diet, num_choices=50):
    # Calculate the caloric needs
    if is_diet:
        calories_needed = (weight - desired_weight) * 7700 / 30  # Simple weight loss estimation
        category = 'Diet'
    else:
        calories_needed = (desired_weight - weight) * 7700 / 30  # Simple weight gain estimation
        category = 'Bulking'

    # Filter beverage data
    filtered_beverages = beverages[(beverages['Category'] == category) & (beverages['Calories (kcal)'] <= calories_needed)].sample(min(num_choices, len(beverages)))

    # Format output as required
    formatted_beverages = []
    for _, row in filtered_beverages.iterrows():
        formatted_item = {
            "Beverage Name": row['Beverage Name'],
            "Ingredients": row['Ingredients'],
            "Additives": row['Additives'],
            "Calories (kcal)": row['Calories (kcal)'],
            "Protein (g)": row['Protein (g)'],
            "Carbohydrates (g)": row['Carbohydrates (g)'],
            "Fats (g)": row['Fats (g)']
        }
        formatted_beverages.append(formatted_item)

    return formatted_beverages

# Define route for food recommendations
@app.route('/food_recommendations', methods=['POST'])
def food_recommendations():
    data = request.json
    height = data['height']
    weight = data['weight']
    desired_weight = data['desired_weight']
    is_diet = data.get('is_diet', False)

    filtered_food = filter_food(height, weight, desired_weight, is_diet)
    
    # JSON response
    return jsonify(filtered_food)

# Define route for beverage recommendations
@app.route('/beverage_recommendations', methods=['POST'])
def beverage_recommendations():
    data = request.json
    height = data['height']
    weight = data['weight']
    desired_weight = data['desired_weight']
    is_diet = data.get('is_diet', False)

    filtered_beverages = filter_beverages(height, weight, desired_weight, is_diet)
    
    # JSON response
    return jsonify(filtered_beverages)

if __name__ == '__main__':
    app.run(debug=True)
