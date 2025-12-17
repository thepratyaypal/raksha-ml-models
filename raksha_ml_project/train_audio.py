import numpy as np
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
            print("\nCRITICAL ERROR: No audio files loaded!")
            print("Please add .wav files to the audio_dataset folders.")
            return None, None
            
        return np.array(X), np.array(y)

os.makedirs("models", exist_ok=True)
p = Proc()
X, y = p.load("audio_dataset")

if X is None or len(X) == 0:
    print("\nTraining aborted: No audio data available")
    print("\nNext steps:")
    print("1. Add .wav files to audio_dataset/scream/")
    print("2. Add .wav files to audio_dataset/panic/")
    print("3. Add .wav files to audio_dataset/normal/")
    print("4. Add .wav files to audio_dataset/noise/")
    print("5. Run: python train_audio.py")
    exit(1)

print("\nTotal loaded: " + str(len(X)) + " samples")

le = LabelEncoder()
ye = le.fit_transform(y)
nc = len(le.classes_)

print("Classes found: " + str(le.classes_.tolist()))

if len(X) < 40:
    print("\nWARNING: Very few samples (" + str(len(X)) + ")")
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

print("\nTraining...")
cb = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
]

m.fit(Xtr, ytr, validation_data=(Xte, yte), epochs=50, batch_size=32, callbacks=cb, verbose=1)

l, a = m.evaluate(Xte, yte, verbose=0)
print("\nFinal Accuracy: " + str(round(a*100, 2)) + "%")

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
