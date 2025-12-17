import numpy as np
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
