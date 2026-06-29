from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import numpy as np
import librosa
import librosa.display
import joblib
import json
from tensorflow.keras.models import load_model
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

# =======================
# Flask Setup
# =======================
app = Flask(__name__)
app.secret_key = "your_secret_key"
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# =======================
# Load Model, Scaler, Classes
# =======================
model = load_model("voice_disease_model.h5")
scaler = joblib.load("scaler.pkl")
with open("classes.json", "r") as f:
    classes = json.load(f)
classes = {int(k): v for k, v in classes.items()}

recommendations = {
    "healthy": "No medical action needed. Maintain vocal hygiene.",
    "laryngitis": "Rest your voice, stay hydrated, avoid shouting. See ENT if persistent.",
    "vocal_nodules": "Consult an ENT or speech therapist for evaluation.",
    "vocal_polyp": "ENT consultation recommended for proper treatment.",
    "laryngeal_cancer": "Immediate medical attention required. Consult an oncologist/ENT specialist.",
    "paralysis": "Consult ENT or neurologist. Voice therapy may help.",
}

# =======================
# Simple User System (replace with DB for production)
# =======================
users = {}

# =======================
# Feature Extraction
# =======================
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=22050)
    max_len = 3 * sr
    if len(audio) < max_len:
        audio = np.pad(audio, (0, max_len - len(audio)))
    else:
        audio = audio[:max_len]

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40, n_fft=512)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=512)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=512)

    features = np.hstack([
        np.mean(mfcc.T, axis=0),
        np.mean(chroma.T, axis=0),
        np.mean(mel.T, axis=0)
    ])
    return features, audio, mfcc, chroma, mel

# =======================
# Prediction & Plots
# =======================
def predict(file_path):
    features, audio, mfcc, chroma, mel = extract_features(file_path)
    features_scaled = scaler.transform([features])
    preds = model.predict(features_scaled)[0]

    idx = np.argmax(preds)
    confidence = np.max(preds) * 100
    disease = classes[idx]
    recommendation_text = recommendations.get(disease, "Consult a specialist for guidance.")

    # Save spectrum
    spectrum_file = os.path.join(UPLOAD_FOLDER, "spectrum.png")
    fig, ax = plt.subplots(figsize=(5,2.5))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    img = librosa.display.specshow(D, sr=22050, x_axis='time', y_axis='log', ax=ax)
    ax.set_title(f"Spectrum: {disease}")
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.savefig(spectrum_file)
    plt.close()

    # Save MFCC
    mfcc_file = os.path.join(UPLOAD_FOLDER, "mfcc.png")
    fig, ax = plt.subplots(figsize=(5,2.5))
    img2 = librosa.display.specshow(mfcc, x_axis='time', ax=ax)
    ax.set_title("MFCC")
    plt.colorbar(img2, ax=ax)
    plt.savefig(mfcc_file)
    plt.close()

    # Save Mel
    mel_file = os.path.join(UPLOAD_FOLDER, "mel.png")
    fig, ax = plt.subplots(figsize=(5,2.5))
    img3 = librosa.display.specshow(librosa.power_to_db(mel, ref=np.max), y_axis='mel', x_axis='time', ax=ax)
    ax.set_title("Mel Spectrogram")
    plt.colorbar(img3, ax=ax, format='%+2.0f dB')
    plt.savefig(mel_file)
    plt.close()

    return disease, confidence, recommendation_text, spectrum_file, mfcc_file, mel_file

# =======================
# Routes
# =======================
@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('index'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('index'))
        else:
            flash("Invalid credentials")
    return render_template('login.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users:
            flash("Username already exists")
        else:
            users[username] = password
            flash("Registration successful! Login now.")
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/index', methods=['GET','POST'])
def index():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'file' not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash("No selected file")
            return redirect(request.url)

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        disease, confidence, recommendation, spectrum_file, mfcc_file, mel_file = predict(file_path)
        return render_template('result.html', disease=disease, confidence=confidence, recommendation=recommendation,
                               spectrum_file=spectrum_file, mfcc_file=mfcc_file, mel_file=mel_file)

    return render_template('index.html', username=session['username'])

# =======================
if __name__ == "__main__":
    app.run(debug=True)