# train.py
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("insurance_claim_approval.csv")

# Drop ID
df = df.drop("claim_id", axis=1)

# Encode 'accident_type'
le = LabelEncoder()
df["accident_type"] = le.fit_transform(df["accident_type"])  # e.g. home = 0, car = 1, etc.

# Split data
X = df.drop("approved", axis=1)
y = df["approved"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(" Accuracy:", accuracy_score(y_test, y_pred))
print(" Classification Report:\n", classification_report(y_test, y_pred))

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")
print(" Model saved at model/model.pkl")
