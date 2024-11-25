import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset
df = pd.read_csv("video_game_store_dataset.csv")

# 2. Define function to categorize Rank into Top, Normal, Low
def categorize_rank(rank):
    if rank <= 10:
        return 'Top Rank'
    elif rank <= 50:
        return 'Normal Rank'
    else:
        return 'Low Rank'

# Apply the categorize_rank function to the Rank column
df['Rank Category'] = df['Rank'].apply(categorize_rank)
  # Assuming categorize_stock function is defined

# Encode categorical columns (Genre, Stock, Rank Category)
label_encoder = LabelEncoder()
df['Genre'] = label_encoder.fit_transform(df['Genre'])  # Encoding the Genre
df['Stock'] = label_encoder.fit_transform(df['Stock'])  # Encoding the Stock
df['Rank Category'] = label_encoder.fit_transform(df['Rank Category'])  # Encoding the Rank Category (Top=0, Normal=1, Low=2)

# 4. Separate features (X) and target (y)
X = df[['Genre', 'Price', 'Rating', 'Stock']]  # Features
y = df['Rank Category']  # Target variable (Rank Category)

# 5. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train the Decision Tree Classifier model
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# 7. Make predictions and evaluate the model
y_pred = dt_classifier.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Optionally, map the numerical predictions back to original labels (Top Rank, Normal Rank, Low Rank)
predictions = label_encoder.inverse_transform(y_pred)
print("Predictions:", predictions)

joblib.dump(model, 'dtree_model.pkl')
