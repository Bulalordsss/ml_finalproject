import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("video_game_store_dataset.csv")

# Features and target variable
X = df[["Rank", "Price", "Rating", "Stock", "Release_Year"]]  # Features
y = df["Sales"]  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display results
print("Linear Regression Model Results:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")

joblib.dump(model, 'linear_regression_model.pkl')
