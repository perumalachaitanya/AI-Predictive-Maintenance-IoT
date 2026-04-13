# STEP 2: DATA LOADING

import pandas as pd

# Load dataset
data = pd.read_csv("data/sensor_data.csv")

# Show first 5 rows
print("📊 Dataset Preview:")
print(data.head())

# Show dataset info
print("\n📌 Dataset Info:")
print(data.info())

# Show basic statistics
print("\n📈 Dataset Statistics:")
print(data.describe())

print("\n❗ Missing Values Check:")
print(data.isnull().sum())

# STEP 3: DATA CLEANING & PREPROCESSING

import pandas as pd

# Load dataset
data = pd.read_csv("data/sensor_data.csv")

print("📊 Original Data:")
print(data.head())

# 1. Check missing values
print("\n❗ Missing Values Before:")
print(data.isnull().sum())

# 2. Handle missing values (if any)
data = data.dropna()

# 3. Remove duplicates
data = data.drop_duplicates()

# 4. Data type check
print("\n📌 Data Types:")
print(data.dtypes)

# 5. Basic cleaning validation
print("\n📊 Cleaned Data Preview:")
print(data.head())

print("\n❗ Missing Values After:")
print(data.isnull().sum())

# Feature scaling (optional for now)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

features = ['temperature', 'vibration', 'current']
data[features] = scaler.fit_transform(data[features])

print("\n⚙️ Scaled Data:")
print(data.head())

# STEP 4: FEATURE ENGINEERING

# Define features (X) and target (y)
X = data[['temperature', 'vibration', 'current']]
y = data['failure']

print("\n📥 Input Features (X):")
print(X.head())

print("\n🎯 Target Variable (y):")
print(y.head())

# Check shapes
print("\n📏 Shape of X:", X.shape)
print("📏 Shape of y:", y.shape)

from sklearn.model_selection import train_test_split

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n✅ Data Split Completed")
print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# STEP 5: MODEL TRAINING

from sklearn.ensemble import RandomForestClassifier

# Create model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train model
model.fit(X_train, y_train)

print("\n🤖 Model Training Completed!")

# Predictions on test data
y_pred = model.predict(X_test)

print("\n🔮 Predictions:")
print(y_pred)

import joblib

# Save model
joblib.dump(model, "models/model.pkl")

print("\n💾 Model saved as model.pkl")

# STEP 6: MODEL EVALUATION

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Precision
precision = precision_score(y_test, y_pred)

# Recall
recall = recall_score(y_test, y_pred)

print("\n📊 Model Evaluation Results:")
print(f"✅ Accuracy: {accuracy:.2f}")
print(f"✅ Precision: {precision:.2f}")
print(f"✅ Recall: {recall:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\n📉 Confusion Matrix:")
print(cm)

# Detailed report
print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred))

# STEP 7: REAL-TIME PREDICTION

print("\n🔮 Real-Time Prediction System")

# Example new sensor data (you can change values)
new_data = pd.DataFrame({
    'temperature': [75],
    'vibration': [4.8],
    'current': [10.2]
})

print("\n📥 New Sensor Input:")
print(new_data)

# IMPORTANT: Apply same scaling used earlier
new_data_scaled = pd.DataFrame(
    scaler.transform(new_data),
    columns=['temperature', 'vibration', 'current']
)

# Predict
prediction = model.predict(new_data_scaled)

# Output result
if prediction[0] == 1:
    print("\n⚠️ ALERT: Machine Failure Predicted!")
else:
    print("\n✅ Machine is Operating Normally")

    # STEP 8: VISUALIZATION

import matplotlib.pyplot as plt

# Create graph folder automatically
import os
os.makedirs("outputs/graphs", exist_ok=True)

# 1. Temperature Graph
plt.figure()
plt.plot(data['temperature'])
plt.title("Temperature Trend")
plt.xlabel("Index")
plt.ylabel("Temperature")
plt.savefig("outputs/graphs/temperature.png")
plt.close()

# 2. Vibration Graph
plt.figure()
plt.plot(data['vibration'])
plt.title("Vibration Trend")
plt.xlabel("Index")
plt.ylabel("Vibration")
plt.savefig("outputs/graphs/vibration.png")
plt.close()

# 3. Current Graph
plt.figure()
plt.plot(data['current'])
plt.title("Current Trend")
plt.xlabel("Index")
plt.ylabel("Current")
plt.savefig("outputs/graphs/current.png")
plt.close()

print("\n📊 Graphs saved in outputs/graphs/")