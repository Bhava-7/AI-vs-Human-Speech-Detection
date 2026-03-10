# AI vs Human Speech Detection Using XGBoost

## Project Overview

This project uses Signal Processing and Machine Learning to classify real vs fake speech. The project leverages audio features like MFCC, Spectral Contrast, Zero Crossing Rate, STFT, Pitch (F0), and Mel Spectrograms to differentiate between human-generated and AI-generated speech. The XGBoost classifier is trained on these features to perform binary classification.

### Features Used:
- MFCC (Mel Frequency Cepstral Coefficients)
- Spectral Contrast
- Zero Crossing Rate (ZCR)
- STFT (Short-Time Fourier Transform)
- Pitch (F0)
- Mel Spectrogram

## Tools Used:
- Python 3.10.11
- Librosa (for audio feature extraction)
- XGBoost (for model training)
- Scikit-learn (for model evaluation and splitting data)
- Joblib (for saving and loading the trained model)
- Matplotlib (for plotting visualizations)
- Pydub (for audio file conversion)

## Getting Started

### Prerequisites
To run the code, you'll need to install the required dependencies. You can do this by running the following command:

## bash
pip install -r requirements.txt

## Dataset

The dataset used for training the model contains both human and AI-generated speech. You can find a similar dataset on Kaggle
.If using your own dataset, ensure it follows the same structure.

## Usage

1. Feature Extraction:
Extract features from the audio files using the feature_extraction.py script.
2. Model Training:
Train the model using the train_model.py script.
3. Prediction:
After training, use the trained model to predict whether an audio file is real or fake.
