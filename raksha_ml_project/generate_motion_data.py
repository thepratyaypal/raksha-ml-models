import numpy as np
import pandas as pd

SAMPLES = 200

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
