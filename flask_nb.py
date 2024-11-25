from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the pre-trained Naive Bayes model
model = joblib.load('model/Naive_Bayes_model.pkl')

@app.route('/')
def home():
    return render_template('naive_bayes.html')  # Link this to your main index page

@app.route('/naive_bayes', methods=['GET', 'POST'])
def naive_bayes():
    if request.method == 'POST':
        # Get the form data
        platform = request.form['platform']  # e.g., 'PC', 'PlayStation', etc.
        price = float(request.form['price'])  # Convert to float
        rating = float(request.form['rating'])  # Convert to float

        # Create a DataFrame with the input data
        input_data = pd.DataFrame({
            'Platform': [platform],
            'Price': [price],
            'Rating': [rating],
            'Release_Year': [2024]  # Use a default or dummy year for simplicity
        })

        try:
            # Make a prediction
            prediction = model.predict(input_data)
            predicted_genre = prediction[0]  # Extract the genre
        except Exception as e:
            return f"An error occurred during prediction: {str(e)}"

        # Render the result in the template
        return render_template('naive_bayes.html', prediction=predicted_genre)

    # If GET request, render the form
    return render_template('naive_bayes.html')

if __name__ == '__main__':
    app.run(debug=True)
