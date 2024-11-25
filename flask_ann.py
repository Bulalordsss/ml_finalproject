from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os

app = Flask(__name__, static_folder='static')

# Load the trained ANN model
model = joblib.load('model/ann_model.pkl')

# Load the encoder and scaler used for preprocessing
encoder = joblib.load('model/encoder.pkl')  # Save this from your training process
scaler = joblib.load('model/scaler.pkl')    # Save this from your training process

# Get the list of game images (assuming images are named 1.png, 2.png, ..., 100.png)
def get_game_images():
    image_paths = [f"img/{i}.png" for i in range(1, 101)]  # List of image paths
    return image_paths

# Route for the form page
@app.route('/')
def index():
    # Get the list of images
    image_paths = get_game_images()  # Make sure it's just the list of image paths
    return render_template('index.html', image_paths=image_paths)

# Route to handle the form submission and make a prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        platform = request.form['platform']
        genre = request.form['genre']
        rating = float(request.form['rating'])
        sales = float(request.form['sales'])

        # Preprocess the input features
        input_data = pd.DataFrame([[platform, genre, rating, sales]], columns=['Platform', 'Genre', 'Rating', 'Sales'])
        
        # One-hot encode categorical features (Platform and Genre)
        categorical_features = input_data[['Genre', 'Platform']]
        encoded_categorical = encoder.transform(categorical_features)

        # Normalize numerical features (Rating and Sales)
        numerical_features = input_data[['Rating', 'Sales']].values
        scaled_numerical = scaler.transform(numerical_features)

        # Combine the encoded categorical features with scaled numerical features
        X_new = np.hstack([encoded_categorical, scaled_numerical])

        # Make prediction using the ANN model
        predicted_price = model.predict(X_new)[0]

        # Get the list of game images
        image_paths = get_game_images()
        return render_template('ann.html', prediction=predicted_price, image_paths=image_paths)

if __name__ == '__main__':
    app.run(debug=True)
