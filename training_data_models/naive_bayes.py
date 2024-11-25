import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
data = pd.read_csv('video_game_store_dataset.csv')

# Features and target variable
X = data[['Platform', 'Price', 'Rating', 'Release_Year']]
y = data['Genre']

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('platform', OneHotEncoder(), ['Platform']),
        ('num', 'passthrough', ['Price', 'Rating', 'Release_Year'])
    ])

# Create a pipeline with preprocessing and model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GaussianNB())
])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f'Model Accuracy: {accuracy:.2f}')

joblib.dump(model, 'Naive_Bayes_model.pkl')