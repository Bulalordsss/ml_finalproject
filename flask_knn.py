from flask import Flask, render_template, request
import joblib
import numpy as np
import os

# Initialize the Flask application
app = Flask(__name__)

# Load the entire pipeline (model + scaler) to ensure consistency
model_path = os.path.join('model', 'knn_pipeline.pkl')
pipeline = joblib.load(model_path)

# If you're using label encoding, load the encoder as well
label_encoder = joblib.load('model/label_encoder.pkl')

@app.route('/')
def index():
    return render_template('knn.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    price = float(request.form['price'])
    sales = float(request.form['sales'])
    rating = float(request.form['rating'])

    # Prepare the input data
    input_data = np.array([[price, sales, rating]])

    # Make prediction using the entire pipeline (scaler + model)
    prediction = pipeline.predict(input_data)

    # Decode the prediction (if you used label encoding in your model)
    stock_label = label_encoder.inverse_transform(prediction)[0]

    return render_template('knn.html', stock_label=stock_label)

if __name__ == '__main__':
    app.run(debug=True)
