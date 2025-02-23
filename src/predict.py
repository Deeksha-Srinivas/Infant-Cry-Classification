import librosa
import numpy as np
import joblib
import os
import sys

# Load the trained model
model_path = "models/cry_classifier.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Trained model not found at {model_path}")

clf = joblib.load(model_path)

# Function to extract features from an audio file
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=5)  # Load audio file (5s)

    # Extract features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    # Compute mean of each feature
    features = np.hstack((
        np.mean(mfcc.T, axis=0),
        np.mean(chroma.T, axis=0),
        np.mean(spectral_contrast.T, axis=0)
    ))

    return features

# Get input audio file from command-line
if len(sys.argv) < 2:
    print("Usage: python predict.py <audio_file.wav>")
    sys.exit(1)

audio_file = sys.argv[1]

if not os.path.exists(audio_file):
    print(f"Error: File '{audio_file}' not found.")
    sys.exit(1)

# Extract features and predict
features = extract_features(audio_file)
features = features.reshape(1, -1)  # Reshape for prediction

prediction = clf.predict(features)[0]

print(f"Predicted Cry Type: {prediction}")
