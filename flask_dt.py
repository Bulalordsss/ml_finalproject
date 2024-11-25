import pandas as pd
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the pre-trained Decision Tree model
dtree_model = joblib.load("model/dtree_model.pkl")

# Categorical mappings (should match the mappings used during training)
genre_mapping = {'Action': 0, 'Adventure': 1, 'Puzzle': 2}  # Adjust as per training

# Reverse mappings for prediction output
rank_mapping = {0: 'Top Rank', 1: 'Normal Rank', 2: 'Low Rank'}  # Adjust as per model

@app.route('/decision_tree', methods=['GET', 'POST'])
def decision_tree():
    prediction = None
    if request.method == 'POST':
        try:
            # Retrieve form data
            genre = request.form['genre']
            price = float(request.form['price'])
            rating = float(request.form['rating'])
            stock = float(request.form['stock'])  # Stock as a float

            # Map input data to numerical encoding
            mapped_genre = genre_mapping.get(genre)

            # Validate input data
            if mapped_genre is None:
                return "Error: Invalid category provided for Genre.", 400

            # Create input DataFrame (include Release_Year)
            input_data = pd.DataFrame([{
                'Genre': mapped_genre,
                'Price': price,
                'Rating': rating,
                'Stock': stock
            }])

            # Make prediction
            rank_prediction = dtree_model.predict(input_data)
            prediction = rank_mapping.get(rank_prediction[0], "Unknown Rank")

        except Exception as e:
            return f"Error: {e}", 500

    # Render the HTML page with prediction
    return render_template('decision_tree.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
