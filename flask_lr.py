from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model from the 'model' folder
model_path = os.path.join('model', 'linear_regression_model.pkl')
model = joblib.load(model_path)

@app.route('/')
def index():
    # Render the HTML form
    return render_template('linear_regression.html')
# For Linear Regression
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form data
        rank = float(request.form['rank'])  # Include Rank
        rating = float(request.form['rating'])
        price = float(request.form['price'])
        stock = float(request.form['stock'])
        release_year = int(request.form['release_year'])

        # Input data for prediction (ensure it has all 5 features)
        input_data = np.array([[rank, price, rating, stock, release_year]])

        # Predict sales using the model
        prediction = model.predict(input_data)[0]

        # Render the same page with the prediction result
        return render_template(
            'linear_regression.html',
            prediction=f"Predicted Sales: {prediction:.2f}"
        )
    except Exception as e:
        # Handle errors and display them on the page
        return render_template(
            'linear_regression.html',
            error=f"An error occurred: {str(e)}"
        )


if __name__ == '__main__':
    app.run(debug=True)
