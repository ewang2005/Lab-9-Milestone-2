import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import data
data = pd.read_csv("WALKING.csv")

#get timestamps
timestamps = data["timestamp"].values
timeSeconds = (timestamps - timestamps[0]) / 1e9

#get accelerometer data
accelX = data["accel_x"].values
accelY = data["accel_y"].values
accelZ = data["accel_z"].values

#get magnitude of acceleration
accelMagnitude = np.sqrt(accelX**2 + accelY**2 + accelZ**2)

#smooth data with moving average of the last 50 observations
windowSize = 50
smoothedAccel = np.convolve(accelMagnitude, np.ones(windowSize)/windowSize, mode='same')

#determine steps
threshold = np.mean(smoothedAccel) + 1 * np.std(smoothedAccel)
steps = []
last_step_time = -1
was_below = True  

for i in range(len(smoothedAccel)):
    if smoothedAccel[i] > threshold and was_below:
        if (timeSeconds[i] - last_step_time) >= 0.3:
            steps.append(i)
            last_step_time = timeSeconds[i]
            was_below = False 
    
    elif smoothedAccel[i] <= threshold:
        was_below = True  

steps = [int(s) for s in steps]

plt.figure(figsize=(12, 6))

#plot raw and smoothed acceleration magnitude
plt.plot(timeSeconds, accelMagnitude, 'b-', alpha=0.3, linewidth=1, label='Raw')
plt.plot(timeSeconds, smoothedAccel, 'r-', linewidth=2.5, label=f'Smoothed (window={windowSize})')
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Acceleration Magnitude (m/s²)', fontsize=12)
plt.title('Raw and Smoothed Acceleration vs Time')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)


#plot steps count graph
plt.figure(figsize=(14, 6))
plt.plot(timeSeconds, smoothedAccel, 'b-', linewidth=2, label='Acceleration')
plt.scatter(timeSeconds[steps], smoothedAccel[steps], color='red', s=100, zorder=3, label=f'Steps (n={len(steps)})')
plt.axhline(y=threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold={threshold}')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s²)')
plt.title("Steps Shown on Smoothed Acceleration Graph")
plt.legend()
plt.grid(True, alpha=0.3)


plt.show()

