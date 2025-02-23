import librosa
import numpy as np
import pandas as pd
import os

# Function to extract features from an audio file
def extract_features(file_path):
    try:
        print(f"Extracting features from: {file_path}")  # Debug print
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
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Define dataset path
dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset"))

# Check if dataset folder exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset folder not found at {dataset_path}")

print(f"Dataset path: {dataset_path}")  # Debug print

# Process all files in dataset
data = []
labels = []

for category in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category)
    
    # Check if it's a directory
    if os.path.isdir(category_path):
        print(f"Processing category: {category}")  # Debug print
        for file_name in os.listdir(category_path):
            file_path = os.path.join(category_path, file_name)
            
            # Process only .wav files
            if file_name.endswith(".wav"):
                feature = extract_features(file_path)
                
                if feature is not None:
                    data.append(feature)
                    labels.append(category)  # Use folder name as label
                else:
                    print(f"Feature extraction failed for: {file_path}")

# Check if features were extracted
if len(data) == 0:
    print("⚠️ No features extracted! Check dataset folder and audio files.")
else:
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(data)
    df["label"] = labels
    df.to_csv("features.csv", index=False)
    print("✅ Feature extraction completed. Data saved to features.csv")