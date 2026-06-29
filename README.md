# AI-Powered Voice Analysis for Medical Diagnosis

## Project Overview

AI-Powered Voice Analysis for Medical Diagnosis is a Flask-based web application that analyzes voice recordings and predicts possible voice-related diseases using Machine Learning and Deep Learning.

## Features

* User Registration and Login
* Upload Voice (.wav) Files
* Voice Disease Prediction
* Confidence Score
* Medical Recommendation
* Voice Signal Analysis using MFCC and Mel Spectrogram

## Technologies Used

* Python
* Flask
* TensorFlow
* Librosa
* NumPy
* Scikit-learn
* HTML
* CSS

## Project Files

* `app.py` – Main Flask application
* `voice_disease_model.h5` – Trained deep learning model
* `scaler.pkl` – Feature scaler
* `classes.json` – Disease class labels
* `recorded.wav` – Sample voice recording

## How to Run

STEP 1: SETUP ENVIRONMENT 
 
• Install Python (3.8+) 
• Open terminal and go to project folder 
 
 
STEP 2: INSTALL DEPENDENCIES 
 
• Run: 
pip install flask numpy librosa tensorflow keras joblib matplotlib scipy 
 
 
STEP 3: ORGANIZE FILES 
Ensure these files are present: 
 
• app.py 
• voice_disease_model.h5 
• scaler.pkl 
• classes.json 
• templates/ (HTML files) 
• static/uploads/ (auto-created or manually create) 
 
 
STEP 4: RUN THE APPLICATION 
 
• Run command: python flaskapp.py 
 
 
STEP 5: OPEN APPLICATION 
• Go to browser Open: http://127.0.0.1:5000 
STEP 6: EXECUTE SYSTEM 
 
• Register and login 
• Upload voice file (.wav) 
• System processes input through all modules 
• Displays prediction, confidence, and graphs 
 
 
STEP 7: VIEW OUTPUT 
 
• Disease name + confidence score 
• Recommendation message 
• Spectrum, MFCC, Mel images 
 
 
STEP 8: TROUBLESHOOT 
 
• Check file paths if model not loading 
• Ensure audio is .wav format 
• Verify all libraries installed 
• Check terminal for errors

