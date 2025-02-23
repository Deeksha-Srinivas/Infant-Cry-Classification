from flask import Flask, request, jsonify
import librosa
import numpy as np
import joblib
from src.feature_extraction import extract_features

app = Flask(__name__)

# Load model and label encoder
model = joblib.load("models/cry_classifier.pkl")
encoder = joblib.load("models/label_encoder.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["audio"]
    
    # Process the uploaded audio file
    y, sr = librosa.load(file, duration=5)
    features = extract_features(y, sr)
    
    # Predict the class
    prediction = model.predict([features])[0]
    predicted_label = encoder.inverse_transform([prediction])[0]
    
    return jsonify({"cry_category": predicted_label})

if __name__ == "__main__":
    app.run(debug=True)
