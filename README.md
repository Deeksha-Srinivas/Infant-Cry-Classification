# Infant Cry Classification using Machine Learning
This project classifies infant cries into categories like hunger, discomfort, tiredness, and pain using Machine Learning & Audio Processing techniques.

# Features
- Extracts audio features using MFCCs, Chroma, and Spectral Contrast
- Trains a Random Forest Classifier for cry classification
- Predicts the cry type from a test audio file
- Implements end-to-end processing, from feature extraction to prediction

# Project Structure
 ```bash

├── dataset/                  # Audio dataset with subfolders
│   ├── hunger/
│   ├── tired/
│   ├── discomfort/
│   ├── ...
│
├── models/                   # Trained model and scaler
│   ├── cry_classifier.pkl    
│   ├── scaler.pkl   
│
├── src/                      # Source code
│   ├── feature_extraction.py # Extracts features from audio files
│   ├── train_model.py        # Trains the ML model
│   ├── predict.py            # Predicts cry type from audio
│
├── requirements.txt          # Dependencies  
├── README.md                 # Project documentation  

```
# Installation & Setup
1 Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```
2️ Extract Features from Audio Files
```bash
python src/feature_extraction.py
```
3️ Train the Model
```bash
python src/train_model.py
```
4️ Predict Cry Type
```bash
python src/predict.py path/to/test_audio.wav
```
# Example Usage
Run the following command to classify a test audio file:
```bash
python src/predict.py dataset/test_audio.wav
```
Output:
Predicted Cry Type: Hungry

# Future Enhancements
🔹 Improve model accuracy with Deep Learning (CNN/RNN)
🔹 Create a Web/Mobile App for real-time predictions
🔹 Expand the dataset for better generalization

# License
This project is open-source and available under the MIT License.

💡Contributions & Feedback are Welcome! 🚀😊
