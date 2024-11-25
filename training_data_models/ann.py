# Import libraries
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = ("video_game_store_dataset.csv")  # Replace with your actual file path
data = pd.read_csv(file_path)

# Step 1: Data Preprocessing
# Selecting relevant features and target variable
features = data[['Genre', 'Platform', 'Rating', 'Sales']]
target = data['Price']

# One-hot encode categorical features (Genre and Platform)
categorical_features = features[['Genre', 'Platform']]
encoder = OneHotEncoder(sparse_output=False)
encoded_categorical = encoder.fit_transform(categorical_features)

# Normalize numerical features
numerical_features = features[['Rating', 'Sales']].values
scaler = StandardScaler()
scaled_numerical = scaler.fit_transform(numerical_features)

# Combine encoded categorical features with normalized numerical features
X = np.hstack([encoded_categorical, scaled_numerical])
y = target

# Step 2: Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Training the ANN model
model = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),  # Increased complexity
    max_iter=1000,  # Increased iterations
    solver='adam',  # Optimized solver
    learning_rate_init=0.01,  # Set initial learning rate
    alpha=0.001,  # L2 regularization
    random_state=42
)
model.fit(X_train, y_train)

# Step 4: Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

joblib.dump(model, 'ann_model.pkl')
