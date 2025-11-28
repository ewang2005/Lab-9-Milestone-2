import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import data
data = pd.read_csv("TURNING.csv")

#get timestamps
timestamps = data["timestamp"].values
timeSeconds = (timestamps - timestamps[0]) / 1e9

#get data
gyroZ = data["gyro_z"].values

#smooth data with moving average
windowSize = 50
smoothedGyroZ = np.convolve(gyroZ, np.ones(windowSize)/windowSize, mode='same')

#calculate threshold
threshold = np.mean(np.abs(smoothedGyroZ)) + 2 * np.std(smoothedGyroZ)

#detect turns
turns = []
turnDirections = []
lastTurnTime = -1
minTurnInterval = 0.5 

for i in range(1, len(smoothedGyroZ) - 1):
    #check if local max or min
    isLocalMax = smoothedGyroZ[i] > smoothedGyroZ[i-1] and smoothedGyroZ[i] > smoothedGyroZ[i+1]
    isLocalMin = smoothedGyroZ[i] < smoothedGyroZ[i-1] and smoothedGyroZ[i] < smoothedGyroZ[i+1]
    
    #check if absolute value is above threshold
    aboveThreshold = abs(smoothedGyroZ[i]) > threshold
    
    #check time since the last turn
    timeOk = (timeSeconds[i] - lastTurnTime) >= minTurnInterval
    
    #if all conditions pass mark as turn
    if (isLocalMax or isLocalMin) and aboveThreshold and timeOk:
        turns.append(int(i))
        
        #determine direction and estimate angle
        #integrate over th turn to get total angle change
        turnWindow = int(0.3 / np.mean(np.diff(timeSeconds))) 
        startIdx = max(0, i - turnWindow)
        endIdx = min(len(smoothedGyroZ), i + turnWindow)
        
        dt = np.mean(np.diff(timeSeconds))
        angleChange = np.sum(smoothedGyroZ[startIdx:endIdx]) * dt
        angleChangeDeg = np.degrees(angleChange)
        
        if smoothedGyroZ[i] > 0:
            turnDirections.append(f"Left ({angleChangeDeg:.0f} deg)")
        else:
            turnDirections.append(f"Right ({angleChangeDeg:.0f} deg)")
        
        lastTurnTime = timeSeconds[i]

numTurns = len(turns)

print(f"Number of turns detected: {numTurns}")

if numTurns > 0:
    print(f"\nTurn details:")
    for i, (turnIdx, direction) in enumerate(zip(turns, turnDirections)):
        print(f"  Turn {i+1}: Time={timeSeconds[turnIdx]:.2f}s, Direction={direction}")



plt.figure(figsize=(12, 6))

#plot raw and smoothed gyroscope data vs time
plt.plot(timeSeconds, gyroZ, 'b-', alpha=0.3, linewidth=1, label='Raw')
plt.plot(timeSeconds, smoothedGyroZ, 'r-', linewidth=2.5, label=f'Smoothed (window={windowSize})')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Angular Velocity (rad/s)', fontsize=12)
plt.title('Raw and Smoothed Gyroscope Z-axis')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)


plt.figure(figsize=(14, 6))

#plot smoothed gyro_z
plt.plot(timeSeconds, smoothedGyroZ, 'b-', linewidth=2, label='Angular Velocity')

#plot turns detected
if numTurns > 0:
    plt.scatter(timeSeconds[turns], smoothedGyroZ[turns], color='red', s=100, zorder=3, edgecolors='black', linewidths=1,label=f'Detected Turns (n={numTurns})')

#plot turn threshold lines
plt.axhline(y=threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold (+-{threshold:.3f} rad/s)')
plt.axhline(y=-threshold, color='green', linestyle='--', linewidth=2)
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Angular Velocity (rad/s)', fontsize=12)
plt.title(f'Turn Detection Results: {numTurns} Turns Detected')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

plt.show()
