import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load the dataset
df = pd.read_csv("video_game_store_dataset.csv")

# 2. Categorize the Stock variable
def categorize_stock(stock):
    if stock <= 20:
        return "Low"
    elif 21 <= stock <= 50:
        return "Medium"
    else:
        return "High"

# Apply the categorization
df['Stock'] = df['Stock'].apply(categorize_stock)

# 3. Preprocess the data
# Encode the target variable (Stock)
label_encoder = LabelEncoder()
df['Stock'] = label_encoder.fit_transform(df['Stock'])  # Low = 0, Medium = 1, High = 2

# Separate features (X) and target (y)
X = df[['Price', 'Sales', 'Rating']]
y = df['Stock']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Train the KNN model
knn = KNeighborsClassifier(n_neighbors=3)  # You can change the number of neighbors
knn.fit(X_train, y_train)

# 6. Make predictions and evaluate the model
y_pred = knn.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Optionally, map the numerical predictions back to original labels (Low, Medium, High)
predictions = label_encoder.inverse_transform(y_pred)
print("Predictions:", predictions)

joblib.dump(model, 'knn_model.pkl')
