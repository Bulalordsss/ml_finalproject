from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
import time
import os

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Load the models
model_lr = joblib.load(os.path.join('model', 'linear_regression_model.pkl'))
model_knn = joblib.load(os.path.join('model', 'knn_pipeline.pkl'))
model_nb = joblib.load(os.path.join('model', 'Naive_Bayes_model.pkl'))
model_svm = joblib.load(os.path.join('model', 'svm_model.pkl'))
model_dtree = joblib.load(os.path.join('model', 'dtree_model.pkl'))
model_ann = joblib.load(os.path.join('model', 'ann_model.pkl'))

# Load preprocessors
encoder = joblib.load('model/encoder.pkl')
scaler = joblib.load('model/scaler.pkl')
label_encoder = joblib.load('model/label_encoder.pkl')
genre_mapping = {'Action': 0, 'Adventure': 1, 'Puzzle': 2}  # Adjust as per training
rank_mapping = {0: 'Top Rank', 1: 'Normal Rank', 2: 'Low Rank'}  # Adjust as per model

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Linear Regression

@app.route('/linear_regression', methods=['GET', 'POST'])
def linear_regression():
    if request.method == 'POST':
        rank = float(request.form['rank'])
        rating = float(request.form['rating'])
        price = float(request.form['price'])
        stock = float(request.form['stock'])
        release_year = int(request.form['release_year'])
        input_data = np.array([[rank, price, rating, stock, release_year]])
        prediction = model_lr.predict(input_data)[0]
        return render_template('linear_regression.html', prediction=f"Predicted Sales: {prediction:.2f}", cache_buster=int(time.time()))
    return render_template('linear_regression.html', cache_buster=int(time.time()))

# KNN
@app.route('/knn', methods=['GET', 'POST'])
def knn():
    if request.method == 'POST':
        try:
            price = float(request.form['price'])
            sales = float(request.form['sales'])
            rating = float(request.form['rating'])
            input_data = np.array([[price, sales, rating]])
            prediction = model_knn.predict(input_data)
            stock_label = label_encoder.inverse_transform(prediction)[0]
            return render_template('knn.html', stock_label=stock_label, cache_buster=int(time.time()))
        except Exception as e:
            return render_template('knn.html', error=str(e), cache_buster=int(time.time()))
    return render_template('knn.html', cache_buster=int(time.time()))

# Naive Bayes
@app.route('/naive_bayes', methods=['GET', 'POST'])
def naive_bayes():
    if request.method == 'POST':
        platform = request.form['platform']
        price = float(request.form['price'])
        rating = float(request.form['rating'])
        input_data = pd.DataFrame({'Platform': [platform], 'Price': [price], 'Rating': [rating], 'Release_Year': [2024]})
        prediction = model_nb.predict(input_data)
        predicted_genre = prediction[0]
        return render_template('naive_bayes.html', prediction=predicted_genre)
    return render_template('naive_bayes.html')

# SVM
@app.route('/svm', methods=['GET', 'POST'])
def svm():
    if request.method == 'POST':
        genre = request.form['genre']
        price = float(request.form['price'])
        sales = float(request.form['sales'])
        release_year = int(request.form['release_year'])
        rating = float(request.form['rating'])
        input_data = pd.DataFrame({'Genre': [genre], 'Price': [price], 'Sales': [sales], 'Release_Year': [release_year], 'Rating': [rating]})
        prediction = model_svm.predict(input_data)
        platform = prediction[0]
        return render_template('svm.html', platform=platform)
    return render_template('svm.html')

# Decision Tree
@app.route('/decision_tree', methods=['GET', 'POST'])
def decision_tree():
    if request.method == 'POST':
        genre = request.form['genre']
        price = float(request.form['price'])
        rating = float(request.form['rating'])
        stock = float(request.form['stock'])
        mapped_genre = genre_mapping.get(genre)
        if mapped_genre is None:
            return "Error: Invalid category provided for Genre.", 400
        input_data = pd.DataFrame([{'Genre': mapped_genre, 'Price': price, 'Rating': rating, 'Stock': stock}])
        rank_prediction = model_dtree.predict(input_data)
        prediction = rank_mapping.get(rank_prediction[0], "Unknown Rank")
        return render_template('decision_tree.html', prediction=prediction)
    return render_template('decision_tree.html')

# ANN
@app.route('/ann', methods=['GET', 'POST'])
def ann():
    if request.method == 'POST':
        platform = request.form['platform']
        genre = request.form['genre']
        rating = float(request.form['rating'])
        sales = float(request.form['sales'])
        input_data = pd.DataFrame([[platform, genre, rating, sales]], columns=['Platform', 'Genre', 'Rating', 'Sales'])
        categorical_features = input_data[['Genre', 'Platform']]
        encoded_categorical = encoder.transform(categorical_features)
        numerical_features = input_data[['Rating', 'Sales']].values
        scaled_numerical = scaler.transform(numerical_features)
        X_new = np.hstack([encoded_categorical, scaled_numerical])
        predicted_price = model_ann.predict(X_new)[0]
        return render_template('ann.html', prediction=predicted_price)
    return render_template('ann.html')

if __name__ == '__main__':
    app.run(debug=True)
