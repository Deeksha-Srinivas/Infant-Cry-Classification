import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Load extracted features
df = pd.read_csv("features.csv")
X = df.drop(columns=["label"])  # Features
y = df["label"]  # Labels

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Print accuracy
accuracy = clf.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# ✅ Ensure 'models/' directory exists
models_dir = "models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Save model
joblib.dump(clf, os.path.join(models_dir, "cry_classifier.pkl"))

print("✅ Model saved successfully in models/cry_classifier.pkl")
