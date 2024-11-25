from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import os

# Initialize the Flask app
app = Flask(__name__)

# Load the trained SVM model
model_path = os.path.join('model', 'svm_model.pkl')
model = joblib.load(model_path)

@app.route('/')
def index():
    return render_template('svm.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    genre = request.form['genre']  # e.g., 'Action'
    price = float(request.form['price'])
    sales = float(request.form['sales'])
    release_year = int(request.form['release_year'])
    rating = float(request.form['rating'])

    # Create a DataFrame to match the expected input format
    input_data = pd.DataFrame({
        'Genre': [genre],
        'Price': [price],
        'Sales': [sales],
        'Release_Year': [release_year],
        'Rating': [rating]  # Add Rating here
    })

    # Predict using the loaded model
    try:
        prediction = model.predict(input_data)
        platform = prediction[0]  # Extract the first (and only) prediction
    except Exception as e:
        return f"An error occurred during prediction: {str(e)}"

    # Return the result
    return render_template('svm.html', platform=platform)


if __name__ == '__main__':
    app.run(debug=True)
