import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# Load the dataset
data = pd.read_csv("forestfires.csv")

# Data Preprocessing
data = pd.get_dummies(data, columns=['month', 'day'], drop_first=True)
data['fire_occurred'] = np.where(data['area'] > 0, 1, 0)  # Binary target variable

# Scale selected features
scaler = StandardScaler()
data[['temp', 'RH', 'wind', 'rain']] = scaler.fit_transform(data[['temp', 'RH', 'wind', 'rain']])

# Define features and target
X = data.drop(['area', 'fire_occurred'], axis=1)
y = data['fire_occurred']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model as a pickle file
with open("forest_fire_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save the training columns to a pickle file
with open("training_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)  # Save the list of feature names used in training

print("Model and training columns have been saved.")
