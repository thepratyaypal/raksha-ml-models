"""
R.A.K.S.H.A Model Testing Demo
Version: 1.0

This script tests your trained models and shows results.
Place this file in raksha_ml_project/ folder and run:
    python demo_test.py

Requirements:
- Trained models in models/ folder
- Test audio files (optional)
"""

import os
import sys
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Check if we're in the right directory
if not os.path.exists("models"):
    print("ERROR: models/ folder not found!")
    print("Make sure you run this from raksha_ml_project/ folder")
    sys.exit(1)

print("="*70)
print("  R.A.K.S.H.A MODEL TESTING DEMO")
print("="*70)

# Import libraries
try:
    import librosa
    import tensorflow as tf
    from tensorflow import keras
    import pandas as pd
    import joblib
    print("\n[OK] All libraries imported successfully")
except ImportError as e:
    print("\n[ERROR] Missing library: " + str(e))
    print("Run: pip install tensorflow librosa pandas scikit-learn joblib")
    sys.exit(1)

# ============================================================================
# PART 1: LOAD AND INSPECT MODELS
# ============================================================================

print("\n" + "="*70)
print("PART 1: LOADING MODELS")
print("="*70)

# Load Audio Model
print("\n[1/4] Loading Audio Model...")
try:
    audio_model = keras.models.load_model("models/audio_model.h5")
    print("  ✓ Audio model loaded successfully")
    print("  - Input shape:", audio_model.input_shape)
    print("  - Output shape:", audio_model.output_shape)
    
    with open("models/audio_metadata.json", "r") as f:
        audio_meta = json.load(f)
    print("  - Classes:", audio_meta["classes"])
    print("  - Training Accuracy: " + str(round(audio_meta["accuracy"]*100, 2)) + "%")
except Exception as e:
    print("  ✗ Error loading audio model:", str(e))
    audio_model = None
    audio_meta = None

# Load Motion Model
print("\n[2/4] Loading Motion Model...")
try:
    motion_model = keras.models.load_model("models/motion_model.h5")
    print("  ✓ Motion model loaded successfully")
    print("  - Input shape:", motion_model.input_shape)
    print("  - Output shape:", motion_model.output_shape)
    
    with open("models/motion_metadata.json", "r") as f:
        motion_meta = json.load(f)
    print("  - Classes:", motion_meta["classes"])
    print("  - Training Accuracy: " + str(round(motion_meta["accuracy"]*100, 2)) + "%")
except Exception as e:
    print("  ✗ Error loading motion model:", str(e))
    motion_model = None
    motion_meta = None

# Load Motion Scaler
print("\n[3/4] Loading Motion Scaler...")
try:
    motion_scaler = joblib.load("models/motion_scaler.pkl")
    print("  ✓ Motion scaler loaded successfully")
except Exception as e:
    print("  ✗ Error loading scaler:", str(e))
    motion_scaler = None

# Check TFLite files
print("\n[4/4] Checking TFLite Models...")
audio_tflite_exists = os.path.exists("models/audio_model.tflite")
motion_tflite_exists = os.path.exists("models/motion_model.tflite")

if audio_tflite_exists:
    size = os.path.getsize("models/audio_model.tflite") / (1024*1024)
    print("  ✓ audio_model.tflite exists (" + str(round(size, 2)) + " MB)")
else:
    print("  ✗ audio_model.tflite NOT FOUND")

if motion_tflite_exists:
    size = os.path.getsize("models/motion_model.tflite") / (1024*1024)
    print("  ✓ motion_model.tflite exists (" + str(round(size, 2)) + " MB)")
else:
    print("  ✗ motion_model.tflite NOT FOUND")

# ============================================================================
# PART 2: TEST AUDIO MODEL
# ============================================================================

print("\n" + "="*70)
print("PART 2: TESTING AUDIO MODEL")
print("="*70)

if audio_model is None:
    print("\n[SKIP] Audio model not loaded")
else:
    # Test with actual files from dataset
    print("\n[Testing with random samples from dataset...]")
    
    categories = ["scream", "panic", "normal", "noise"]
    test_results = []
    
    for category in categories:
        folder_path = os.path.join("audio_dataset", category)
        if not os.path.exists(folder_path):
            print(f"\n  [SKIP] {category}/ folder not found")
            continue
        
        wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        if len(wav_files) == 0:
            print(f"\n  [SKIP] No .wav files in {category}/")
            continue
        
        # Test first 3 files
        print(f"\n  Testing {category.upper()} samples:")
        for i in range(min(3, len(wav_files))):
            file_path = os.path.join(folder_path, wav_files[i])
            
            try:
                # Load and process audio
                audio, sr = librosa.load(file_path, sr=22050, duration=3)
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
                
                # Pad or truncate to match training
                target_len = int(22050 * 3 / 512) + 1
                if mfcc.shape[1] < target_len:
                    mfcc = np.pad(mfcc, ((0, 0), (0, target_len - mfcc.shape[1])), mode='constant')
                else:
                    mfcc = mfcc[:, :target_len]
                
                # Reshape for model
                mfcc_input = mfcc.reshape(1, mfcc.shape[0], mfcc.shape[1])
                
                # Predict
                prediction = audio_model.predict(mfcc_input, verbose=0)
                predicted_class = audio_meta["classes"][np.argmax(prediction)]
                confidence = np.max(prediction) * 100
                
                # Display result
                is_correct = predicted_class == category
                status = "✓ CORRECT" if is_correct else "✗ WRONG"
                print(f"    {wav_files[i][:30]:30} → {predicted_class:8} ({confidence:5.1f}%) {status}")
                
                test_results.append({
                    "category": category,
                    "predicted": predicted_class,
                    "correct": is_correct,
                    "confidence": confidence
                })
                
            except Exception as e:
                print(f"    {wav_files[i][:30]:30} → ERROR: {str(e)}")
    
    # Summary
    if test_results:
        correct = sum(1 for r in test_results if r["correct"])
        total = len(test_results)
        accuracy = (correct / total) * 100
        avg_confidence = np.mean([r["confidence"] for r in test_results])
        
        print("\n" + "-"*70)
        print(f"  AUDIO MODEL TEST SUMMARY:")
        print(f"  - Tested: {total} samples")
        print(f"  - Correct: {correct}/{total}")
        print(f"  - Accuracy: {accuracy:.1f}%")
        print(f"  - Avg Confidence: {avg_confidence:.1f}%")
        print("-"*70)

# ============================================================================
# PART 3: TEST MOTION MODEL
# ============================================================================

print("\n" + "="*70)
print("PART 3: TESTING MOTION MODEL")
print("="*70)

if motion_model is None or motion_scaler is None:
    print("\n[SKIP] Motion model not loaded")
else:
    print("\n[Testing with synthetic motion data...]")
    
    # Load motion dataset
    if os.path.exists("motion_dataset.csv"):
        df = pd.read_csv("motion_dataset.csv")
        print(f"  ✓ Loaded motion dataset: {len(df)} rows")
        
        # Test random samples from each category
        motion_categories = df['label'].unique()
        motion_results = []
        
        for category in motion_categories:
            category_data = df[df['label'] == category]
            
            # Take first sequence (50 samples)
            if len(category_data) >= 50:
                sample = category_data.iloc[:50]
                
                # Prepare features
                features = sample[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']].values
                features_scaled = motion_scaler.transform(features)
                
                # Reshape for model
                motion_input = features_scaled.reshape(1, 50, 6)
                
                # Predict
                prediction = motion_model.predict(motion_input, verbose=0)
                predicted_class = motion_meta["classes"][np.argmax(prediction)]
                confidence = np.max(prediction) * 100
                
                # Display result
                is_correct = predicted_class == category
                status = "✓ CORRECT" if is_correct else "✗ WRONG"
                print(f"  {category:10} → {predicted_class:10} ({confidence:5.1f}%) {status}")
                
                motion_results.append({
                    "category": category,
                    "predicted": predicted_class,
                    "correct": is_correct,
                    "confidence": confidence
                })
        
        # Summary
        if motion_results:
            correct = sum(1 for r in motion_results if r["correct"])
            total = len(motion_results)
            accuracy = (correct / total) * 100
            avg_confidence = np.mean([r["confidence"] for r in motion_results])
            
            print("\n" + "-"*70)
            print(f"  MOTION MODEL TEST SUMMARY:")
            print(f"  - Tested: {total} categories")
            print(f"  - Correct: {correct}/{total}")
            print(f"  - Accuracy: {accuracy:.1f}%")
            print(f"  - Avg Confidence: {avg_confidence:.1f}%")
            print("-"*70)
    else:
        print("  [SKIP] motion_dataset.csv not found")

# ============================================================================
# PART 4: DEPLOYMENT READINESS CHECK
# ============================================================================

print("\n" + "="*70)
print("PART 4: DEPLOYMENT READINESS CHECK")
print("="*70)

checks = []

# Check 1: Models exist
checks.append({
    "name": "Audio model (.h5) exists",
    "status": audio_model is not None,
    "required": True
})

checks.append({
    "name": "Motion model (.h5) exists",
    "status": motion_model is not None,
    "required": True
})

# Check 2: TFLite models exist
checks.append({
    "name": "Audio TFLite model exists",
    "status": audio_tflite_exists,
    "required": True
})

checks.append({
    "name": "Motion TFLite model exists",
    "status": motion_tflite_exists,
    "required": True
})

# Check 3: Metadata exists
checks.append({
    "name": "Audio metadata exists",
    "status": audio_meta is not None,
    "required": True
})

checks.append({
    "name": "Motion metadata exists",
    "status": motion_meta is not None,
    "required": True
})

# Check 4: Model accuracy
if audio_meta:
    checks.append({
        "name": "Audio accuracy > 70%",
        "status": audio_meta["accuracy"] > 0.70,
        "required": False,
        "value": f"{audio_meta['accuracy']*100:.1f}%"
    })

if motion_meta:
    checks.append({
        "name": "Motion accuracy > 70%",
        "status": motion_meta["accuracy"] > 0.70,
        "required": False,
        "value": f"{motion_meta['accuracy']*100:.1f}%"
    })

# Check 5: TFLite file sizes
if audio_tflite_exists:
    size = os.path.getsize("models/audio_model.tflite") / (1024*1024)
    checks.append({
        "name": "Audio TFLite < 10MB",
        "status": size < 10,
        "required": False,
        "value": f"{size:.2f}MB"
    })

if motion_tflite_exists:
    size = os.path.getsize("models/motion_model.tflite") / (1024*1024)
    checks.append({
        "name": "Motion TFLite < 10MB",
        "status": size < 10,
        "required": False,
        "value": f"{size:.2f}MB"
    })

# Display checks
print("\nChecking deployment requirements:\n")
for check in checks:
    status = "✓ PASS" if check["status"] else "✗ FAIL"
    required = "[REQUIRED]" if check.get("required", False) else "[OPTIONAL]"
    value = f" ({check['value']})" if "value" in check else ""
    print(f"  {status} {required:12} {check['name']}{value}")

# Final verdict
required_checks = [c for c in checks if c.get("required", False)]
all_required_pass = all(c["status"] for c in required_checks)

print("\n" + "="*70)
if all_required_pass:
    print("  ✓ READY FOR DEPLOYMENT!")
    print("\n  Next steps:")
    print("  1. Copy models/audio_model.tflite to your React Native app")
    print("  2. Copy models/motion_model.tflite to your React Native app")
    print("  3. Place them in: android/app/src/main/assets/")
    print("  4. Integrate inference code")
    print("  5. Test on real device")
else:
    print("  ✗ NOT READY - Missing required components")
    print("\n  Fix these issues:")
    for check in required_checks:
        if not check["status"]:
            print(f"    - {check['name']}")
print("="*70)

# ============================================================================
# PART 5: QUICK START GUIDE
# ============================================================================

print("\n" + "="*70)
print("PART 5: QUICK START - REACT NATIVE INTEGRATION")
print("="*70)

integration_code = """
// Step 1: Install dependencies
npm install @tensorflow/tfjs
npm install @tensorflow/tfjs-react-native
npm install react-native-fs

// Step 2: Copy models to android/app/src/main/assets/

// Step 3: Load models in your React Native app
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';
import RNFS from 'react-native-fs';

async function loadModels() {
  await tf.ready();
  
  // Load audio model
  const audioModelPath = RNFS.MainBundlePath + '/audio_model.tflite';
  const audioModel = await tf.loadLayersModel(audioModelPath);
  
  // Load motion model
  const motionModelPath = RNFS.MainBundlePath + '/motion_model.tflite';
  const motionModel = await tf.loadLayersModel(motionModelPath);
  
  console.log('Models loaded!');
  return { audioModel, motionModel };
}

// Step 4: Use models for prediction
async function detectDistress(audioData, motionData) {
  const { audioModel, motionModel } = await loadModels();
  
  // Process audio
  const audioPrediction = audioModel.predict(audioData);
  const audioClass = audioPrediction.argMax(-1).dataSync()[0];
  
  // Process motion
  const motionPrediction = motionModel.predict(motionData);
  const motionClass = motionPrediction.argMax(-1).dataSync()[0];
  
  return { audioClass, motionClass };
}
"""

print(integration_code)

print("\n" + "="*70)
print("  DEMO COMPLETE!")
print("="*70)
print("\nFor detailed integration guide, check React Native TensorFlow docs:")
print("https://github.com/tensorflow/tfjs/tree/master/tfjs-react-native")
print("\n" + "="*70)