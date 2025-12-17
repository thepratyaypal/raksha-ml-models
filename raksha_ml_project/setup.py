"""
R.A.K.S.H.A - Setup with Manual Audio Collection
Version: 2.1 (Manual Audio + Auto Motion)

Usage: python setup_raksha.py

This will:
1. Create project folders for YOU to add audio files
2. Install required packages
3. Generate motion training data automatically
4. Train both AI models (after you add audio)
"""

import os
import sys
import subprocess

# Config
PROJECT = "raksha_ml_project"
MOTION_SAMPLES = 200

def header(text):
    print("\n" + "="*60)
    print("  " + text)
    print("="*60)

def count_audio_files():
    """Count audio files in each category"""
    counts = {}
    categories = ["scream", "panic", "normal", "noise"]
    for cat in categories:
        path = os.path.join("audio_dataset", cat)
        if os.path.exists(path):
            wav_files = [f for f in os.listdir(path) if f.endswith('.wav')]
            counts[cat] = len(wav_files)
        else:
            counts[cat] = 0
    return counts

def main():
    header("R.A.K.S.H.A AUTOMATED SETUP (MANUAL AUDIO)")
    
    print("\nSETUP OVERVIEW:")
    print("  - Create " + PROJECT + "/")
    print("  - Install ML packages")
    print("  - YOU add audio files manually")
    print("  - Generate " + str(MOTION_SAMPLES) + " motion samples automatically")
    print("  - Train 2 AI models")
    print("\nTime needed: 30-60 minutes (after audio collection)")
    
    answer = input("\nStart setup? (yes/no): ").lower()
    
    if answer not in ["yes", "y"]:
        print("Cancelled.")
        return
    
    # STEP 1: Folders
    header("STEP 1: Creating Folders")
    try:
        os.makedirs(PROJECT, exist_ok=True)
        print("Created: " + PROJECT)
        os.chdir(PROJECT)
        os.makedirs("models", exist_ok=True)
        print("Created: models/")
        os.makedirs("audio_dataset/scream", exist_ok=True)
        os.makedirs("audio_dataset/panic", exist_ok=True)
        os.makedirs("audio_dataset/normal", exist_ok=True)
        os.makedirs("audio_dataset/noise", exist_ok=True)
        print("Created: audio_dataset/ with 4 subfolders")
        print("\nSUCCESS!")
    except Exception as e:
        print("ERROR: " + str(e))
        return
    
    # STEP 2: Check Python
    header("STEP 2: Checking Python")
    v = sys.version_info
    print("Python version: " + str(v.major) + "." + str(v.minor))
    if v.major < 3 or v.minor < 8:
        print("ERROR: Need Python 3.8+")
        return
    print("SUCCESS!")
    
    # STEP 3: Install packages
    header("STEP 3: Installing Packages")
    print("This takes 5-10 minutes...\n")
    
    packages = [
        "tensorflow==2.15.0",
        "numpy", "pandas", "matplotlib",
        "scikit-learn", "librosa",
        "soundfile", "scipy", "joblib"
    ]
    
    for pkg in packages:
        print("Installing: " + pkg)
        cmd = sys.executable + " -m pip install " + pkg + " --quiet"
        try:
            subprocess.run(cmd, shell=True, check=True, capture_output=True)
            print("  OK")
        except:
            print("  WARNING: May have issues, continuing...")
    
    print("\nSUCCESS!")
    
    # STEP 4: Generate scripts
    header("STEP 4: Creating Training Scripts")
    
    # Motion data generation script (UNCHANGED)
    motion_data_code = """import numpy as np
import pandas as pd

SAMPLES = """ + str(MOTION_SAMPLES) + """

print("Generating motion dataset...")
data = []

def gn():
    t = np.linspace(0, 2, 50)
    ax = 0.2 * np.sin(2 * np.pi * t) + np.random.normal(0, 0.05, 50)
    ay = 0.3 * np.sin(2 * np.pi * 1.2 * t) + 9.8 + np.random.normal(0, 0.05, 50)
    az = 0.1 * np.sin(2 * np.pi * 0.8 * t) + np.random.normal(0, 0.05, 50)
    gx = np.random.normal(0, 0.1, 50)
    gy = np.random.normal(0, 0.1, 50)
    gz = np.random.normal(0, 0.1, 50)
    return np.column_stack([t, ax, ay, az, gx, gy, gz])

def gf():
    t = np.linspace(0, 2, 50)
    ay = np.ones(50) * 9.8
    ay[20:30] = -15
    ax = np.random.normal(0, 1, 50)
    az = np.random.normal(0, 1, 50)
    gx = np.random.normal(0, 2, 50)
    gy = np.random.normal(0, 2, 50)
    gz = np.random.normal(0, 2, 50)
    gx[20:30] *= 5
    return np.column_stack([t, ax, ay, az, gx, gy, gz])

def gs():
    t = np.linspace(0, 2, 50)
    f = 5
    ax = 5 * np.sin(2 * np.pi * f * t)
    ay = 5 * np.sin(2 * np.pi * (f + 0.5) * t) + 9.8
    az = 5 * np.sin(2 * np.pi * (f - 0.3) * t)
    gx = 8 * np.sin(2 * np.pi * f * t)
    gy = 8 * np.sin(2 * np.pi * (f + 0.7) * t)
    gz = 8 * np.sin(2 * np.pi * (f - 0.4) * t)
    return np.column_stack([t, ax, ay, az, gx, gy, gz])

def gr():
    t = np.linspace(0, 2, 50)
    f = 2.5
    ax = 1.5 * np.sin(2 * np.pi * f * t)
    ay = 2.0 * np.sin(2 * np.pi * (f + 0.2) * t) + 9.8
    az = 1.0 * np.sin(2 * np.pi * (f - 0.1) * t)
    gx = 0.5 * np.sin(2 * np.pi * f * t)
    gy = 0.5 * np.sin(2 * np.pi * (f + 0.3) * t)
    gz = 0.3 * np.sin(2 * np.pi * (f - 0.2) * t)
    return np.column_stack([t, ax, ay, az, gx, gy, gz])

mg = {"normal": gn, "fall": gf, "shake": gs, "running": gr}

for cls in mg:
    print("  " + cls + ": " + str(SAMPLES) + " sequences")
    for _ in range(SAMPLES):
        seq = mg[cls]()
        for r in seq:
            data.append({"timestamp": r[0], "acc_x": r[1], "acc_y": r[2], "acc_z": r[3], "gyro_x": r[4], "gyro_y": r[5], "gyro_z": r[6], "label": cls})

df = pd.DataFrame(data)
df.to_csv("motion_dataset.csv", index=False)
print("Done! Created " + str(len(df)) + " rows")
"""
    
    with open("generate_motion_data.py", "w", encoding="utf-8") as f:
        f.write(motion_data_code)
    print("Created: generate_motion_data.py")
    
    # Audio training script (MODIFIED FOR MANUAL FILES)
    audio_code = """import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os, json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("="*50)
print("Training Audio Model (Manual Files)")
print("="*50)

class Proc:
    def __init__(self):
        self.sr = 22050
        self.dur = 3
        self.mfcc = 40
        self.mlen = int(self.sr * self.dur / 512) + 1
    
    def ext(self, path):
        try:
            au, sr = librosa.load(path, sr=self.sr, duration=self.dur)
            if len(au) == 0:
                print("  WARNING: Empty audio: " + path)
                return None
            mf = librosa.feature.mfcc(y=au, sr=sr, n_mfcc=self.mfcc)
            if mf.shape[1] < self.mlen:
                mf = np.pad(mf, ((0, 0), (0, self.mlen - mf.shape[1])), mode='constant')
            else:
                mf = mf[:, :self.mlen]
            return mf
        except Exception as e:
            print("  ERROR loading " + path + ": " + str(e))
            return None
    
    def load(self, d):
        X, y = [], []
        categories = ["scream", "panic", "normal", "noise"]
        
        for c in categories:
            p = os.path.join(d, c)
            if not os.path.isdir(p):
                print("ERROR: Folder not found: " + p)
                continue
            
            wav_files = [f for f in os.listdir(p) if f.endswith('.wav')]
            
            if len(wav_files) == 0:
                print("ERROR: No .wav files in " + c + "/ folder!")
                print("   Please add audio files before training.")
                return None, None
            
            print("Loading " + c + ": " + str(len(wav_files)) + " files...")
            loaded = 0
            for f in wav_files:
                mf = self.ext(os.path.join(p, f))
                if mf is not None:
                    X.append(mf)
                    y.append(c)
                    loaded += 1
            
            print("  Successfully loaded " + str(loaded) + " files")
            
            if loaded < 20:
                print("  WARNING: Only " + str(loaded) + " files in " + c)
                print("     Recommend at least 50 files per category for good accuracy")
        
        if len(X) == 0:
            print("\\nCRITICAL ERROR: No audio files loaded!")
            print("Please add .wav files to the audio_dataset folders.")
            return None, None
            
        return np.array(X), np.array(y)

os.makedirs("models", exist_ok=True)
p = Proc()
X, y = p.load("audio_dataset")

if X is None or len(X) == 0:
    print("\\nTraining aborted: No audio data available")
    print("\\nNext steps:")
    print("1. Add .wav files to audio_dataset/scream/")
    print("2. Add .wav files to audio_dataset/panic/")
    print("3. Add .wav files to audio_dataset/normal/")
    print("4. Add .wav files to audio_dataset/noise/")
    print("5. Run: python train_audio.py")
    exit(1)

print("\\nTotal loaded: " + str(len(X)) + " samples")

le = LabelEncoder()
ye = le.fit_transform(y)
nc = len(le.classes_)

print("Classes found: " + str(le.classes_.tolist()))

if len(X) < 40:
    print("\\nWARNING: Very few samples (" + str(len(X)) + ")")
    print("Training will continue but accuracy may be low.")
    print("Recommend at least 200 total samples (50 per class)")

Xtr, Xte, ytr, yte = train_test_split(X, ye, test_size=0.2, random_state=42, stratify=ye)

m = keras.Sequential([
    layers.Input(shape=(Xtr.shape[1], Xtr.shape[2])),
    layers.Reshape((Xtr.shape[1], Xtr.shape[2], 1)),
    layers.Conv2D(32, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2),
    layers.Dropout(0.25),
    layers.Conv2D(64, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2),
    layers.Dropout(0.25),
    layers.Conv2D(128, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(nc, activation='softmax')
])

m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("\\nTraining...")
cb = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
]

m.fit(Xtr, ytr, validation_data=(Xte, yte), epochs=50, batch_size=32, callbacks=cb, verbose=1)

l, a = m.evaluate(Xte, yte, verbose=0)
print("\\nFinal Accuracy: " + str(round(a*100, 2)) + "%")

m.save("models/audio_model.h5")

md = {"classes": le.classes_.tolist(), "accuracy": float(a)}
with open("models/audio_metadata.json", "w") as f:
    json.dump(md, f)

c = tf.lite.TFLiteConverter.from_keras_model(m)
c.optimizations = [tf.lite.Optimize.DEFAULT]
tfl = c.convert()
with open("models/audio_model.tflite", "wb") as f:
    f.write(tfl)

print("Saved models!")
print("   - models/audio_model.h5")
print("   - models/audio_model.tflite")
print("   - models/audio_metadata.json")
"""
    
    with open("train_audio.py", "w", encoding="utf-8") as f:
        f.write(audio_code)
    print("Created: train_audio.py")
    
    # Motion training script (UNCHANGED)
    motion_code = """import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os, json, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*50)
print("Training Motion Model")
print("="*50)

class Proc:
    def __init__(self):
        self.ws = 50
        self.sc = StandardScaler()
    
    def win(self, d, l):
        w, wl = [], []
        for i in range(0, len(d) - self.ws, 25):
            w.append(d[i:i + self.ws])
            wl.append(l[i + self.ws // 2])
        return np.array(w), np.array(wl)
    
    def load(self, f):
        df = pd.read_csv(f)
        fc = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        X = df[fc].values
        y = df['label'].values
        print("Loaded " + str(len(X)) + " rows")
        Xs = self.sc.fit_transform(X)
        return self.win(Xs, y)

p = Proc()
X, y = p.load("motion_dataset.csv")
print("Created " + str(len(X)) + " windows")

le = LabelEncoder()
ye = le.fit_transform(y)
nc = len(le.classes_)

Xtr, Xte, ytr, yte = train_test_split(X, ye, test_size=0.2, random_state=42, stratify=ye)

m = keras.Sequential([
    layers.Input(shape=(Xtr.shape[1], Xtr.shape[2])),
    layers.Conv1D(64, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(2),
    layers.Dropout(0.3),
    layers.Conv1D(128, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(2),
    layers.Dropout(0.3),
    layers.Conv1D(256, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.GlobalMaxPooling1D(),
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(nc, activation='softmax')
])

m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Training...")
cb = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7)
]

m.fit(Xtr, ytr, validation_data=(Xte, yte), epochs=100, batch_size=64, callbacks=cb, verbose=1)

l, a = m.evaluate(Xte, yte, verbose=0)
print("Accuracy: " + str(round(a*100, 2)) + "%")

m.save("models/motion_model.h5")

md = {"classes": le.classes_.tolist(), "accuracy": float(a)}
with open("models/motion_metadata.json", "w") as f:
    json.dump(md, f)

joblib.dump(p.sc, "models/motion_scaler.pkl")

c = tf.lite.TFLiteConverter.from_keras_model(m)
c.optimizations = [tf.lite.Optimize.DEFAULT]
tfl = c.convert()
with open("models/motion_model.tflite", "wb") as f:
    f.write(tfl)

print("Saved models!")
"""
    
    with open("train_motion.py", "w", encoding="utf-8") as f:
        f.write(motion_code)
    print("Created: train_motion.py")
    
    print("\nSUCCESS!")
    
    # STEP 5: Generate motion data only
    header("STEP 5: Generating Motion Data")
    print("Creating motion dataset CSV...")
    result = subprocess.run([sys.executable, "generate_motion_data.py"])
    if result.returncode != 0:
        print("ERROR in motion data generation!")
        return
    print("SUCCESS!")
    
    # STEP 6: Audio collection instructions
    header("PAUSE: ADD YOUR AUDIO FILES")
    print("\nPlease add your .wav audio files to these folders:")
    print("   " + os.getcwd() + os.sep + "audio_dataset" + os.sep)
    print("\nFolder structure:")
    print("   audio_dataset/")
    print("   ├── scream/     <- Add distress scream recordings here")
    print("   ├── panic/      <- Add panic/fear sounds here")
    print("   ├── normal/     <- Add normal speech/sounds here")
    print("   └── noise/      <- Add background noise here")
    print("\nRequirements:")
    print("   - Format: .wav files only")
    print("   - Duration: 2-4 seconds per clip")
    print("   - Minimum: 50 files per folder (200 total)")
    print("   - Recommended: 100+ files per folder (400+ total)")
    
    # Check current status
    counts = count_audio_files()
    print("\nCurrent status:")
    total = 0
    all_good = True
    for cat, count in counts.items():
        status = "OK" if count >= 50 else "NEED MORE"
        print("   [" + status + "] " + cat + ": " + str(count) + " files")
        total += count
        if count < 50:
            all_good = False
    print("   Total: " + str(total) + " files")
    
    if total < 40:
        print("\nYou need to add audio files before training!")
        print("\nAfter adding files, run:")
        print("   python train_audio.py")
        print("   python train_motion.py")
        return
    
    if not all_good:
        print("\nWARNING: Some categories have less than 50 files")
        print("Training may have lower accuracy")
        cont = input("\nContinue anyway? (yes/no): ").lower()
        if cont not in ["yes", "y"]:
            print("\nSetup paused. Add more files and run:")
            print("   python train_audio.py")
            print("   python train_motion.py")
            return
    
    print("\nEnough files found! Continuing with training...")
    
    # STEP 7: Train audio
    header("STEP 6: Training Audio Model (15-20 mins)")
    result = subprocess.run([sys.executable, "train_audio.py"])
    if result.returncode != 0:
        print("ERROR in audio training!")
        return
    print("SUCCESS!")
    
    # STEP 8: Train motion
    header("STEP 7: Training Motion Model (20-30 mins)")
    result = subprocess.run([sys.executable, "train_motion.py"])
    if result.returncode != 0:
        print("ERROR in motion training!")
        return
    print("SUCCESS!")
    
    # Done!
    header("COMPLETE!")
    print("\nGenerated files:")
    print("   - models/audio_model.tflite")
    print("   - models/motion_model.tflite")
    print("   - models/audio_metadata.json")
    print("   - models/motion_metadata.json")
    print("\nNext step:")
    print("   Copy the .tflite files to your React Native app!")
    print("\n" + "="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
    except Exception as e:
        print("\nERROR: " + str(e))
        import traceback
        traceback.print_exc()