import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ✅ 1. Load dataset
df = pd.read_csv("E:\\project\\heartattackpredictor\\data\\heartattack.csv")  
df.fillna(df.median(numeric_only=True), inplace=True)

print(df.head())
print(df.info())


# ✅ 2. Define features & target
features = [
    "age", "currentSmoker", "cigsPerDay", "BPMeds", "prevalentHyp",
    "diabetes", "totChol", "sysBP", "diaBP", "BMI", "glucose"
]

target = "TenYearCHD"


# ✅ 3. Split X and y
X = df[features]
y = df[target]

print("Training on all features:", features)


# ✅ 4. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# scaler for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ✅ 5. Models
log_model = LogisticRegression(max_iter=200)
dt_model = DecisionTreeClassifier(max_depth=6)

log_model.fit(X_train_scaled, y_train)
dt_model.fit(X_train, y_train)

log_acc = log_model.score(X_test_scaled, y_test)
dt_acc = dt_model.score(X_test, y_test)

print("\nLogistic Regression Accuracy:", log_acc)
print("Decision Tree Accuracy:", dt_acc)


# Choose model
final_model = log_model
print("\n✔ Logistic Regression selected as final model")


# Ensure folder exists
os.makedirs("models", exist_ok=True)


# Save model + scaler
joblib.dump(final_model, "models/final_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✔ Model and scaler saved in /models folder")


# Test prediction
sample_input = [[45, 1, 10, 0, 1, 0, 220, 130, 85, 26, 90]]
sample_scaled = scaler.transform(sample_input)
pred = final_model.predict(sample_scaled)

print("Sample prediction:", pred)
