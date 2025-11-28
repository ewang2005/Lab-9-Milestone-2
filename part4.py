import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

csv_file = "WALKING_AND_TURNING.csv"

with open(csv_file, 'r') as f:
    lines = f.readlines()

#remove trailing commas
cleaned_lines = [line.rstrip().rstrip(',') + '\n' for line in lines]

#join lines into single string
cleaned_csv = ''.join(cleaned_lines)


#load cleaned data
from io import StringIO
data = pd.read_csv(StringIO(cleaned_csv))

#get timestamps
timestamps = data['timestamp'].values
timeSeconds = (timestamps - timestamps[0]) / 1e9

#get acceleration values
accelX = data['accel_x'].values
accelY = data['accel_y'].values
accelZ = data['accel_z'].values
accelMagnitude = np.sqrt(accelX**2 + accelY**2 + accelZ**2)

#smooth
windowSize = 50
smoothedAccel = np.convolve(accelMagnitude, np.ones(windowSize)/windowSize, mode='same')

#set threshold to be one standard deviation above the mean
threshold = np.mean(smoothedAccel) + np.std(smoothedAccel)

#count steps
steps = []
lastStepTime = -1

for i in range(1, len(smoothedAccel) - 1):
    isLocalMax = smoothedAccel[i] > smoothedAccel[i-1] and smoothedAccel[i] > smoothedAccel[i+1]
    aboveThreshold = smoothedAccel[i] > threshold
    timeOk = (timeSeconds[i] - lastStepTime) >= 0.3
    
    if isLocalMax and aboveThreshold and timeOk:
        steps.append(int(i))
        lastStepTime = timeSeconds[i]

steps = np.array(steps, dtype=int)


#load gyro data to track turns
gyroZ = data['gyro_z'].values

#smooth
windowSizeGyro = 50
smoothedGyroZ = np.convolve(gyroZ, np.ones(windowSizeGyro)/windowSizeGyro, mode='same')

#get angle by integration
dt = np.mean(np.diff(timeSeconds))
cumulativeAngle = np.cumsum(smoothedGyroZ * dt)
cumulativeAngleDeg = np.degrees(cumulativeAngle)

totalRotation = cumulativeAngleDeg.max() - cumulativeAngleDeg.min()

print(f"  Gyro range: {gyroZ.min():.2f} to {gyroZ.max():.2f} rad/s")
print(f"  Total rotation: {totalRotation:.1f}°")

#detect turns of at least 30 degrees
turnThreshold = 30 
minTurnInterval = 0.5
windowSamples = int(0.4 / dt)

turns = []
turnAngles = []
lastTurnTime = -1

for i in range(windowSamples, len(cumulativeAngleDeg) - windowSamples):
    if (timeSeconds[i] - lastTurnTime) < minTurnInterval:
        continue
    
    startIdx = i - windowSamples
    endIdx = i + windowSamples
    angleChange = cumulativeAngleDeg[endIdx] - cumulativeAngleDeg[startIdx]
    
    if abs(angleChange) > turnThreshold:
        #round turns to the nearest 45 degrees
        roundedAngle = round(angleChange / 45) * 45
        
        if roundedAngle != 0:
            turns.append(i)
            turnAngles.append(roundedAngle)
            lastTurnTime = timeSeconds[i]

turns = np.array(turns, dtype=int)
turnAngles = np.array(turnAngles)

#assume step length is 1 meter and we are facing north 
stepLength = 1.0  
startHeading = 90 

#assume start position at (0, 0)
currentX = 0.0
currentY = 0.0
currentHeading = startHeading

pathX = [currentX]
pathY = [currentY]

turnX = []
turnY = []
turnLabels = []

#combine the turns and steps in chronological order using timestamps
events = []
for i in range(len(steps)):
    events.append(("step", timeSeconds[steps[i]], i))
for i in range(len(turns)):
    events.append(("turn", timeSeconds[turns[i]], i))

events.sort(key=lambda x: x[1])

#simulate walking
for eventType, eventTime, eventIdx in events:
    if eventType == "step":
        #move forward one meter
        headingRad = np.radians(currentHeading)
        currentX += stepLength * np.cos(headingRad)
        currentY += stepLength * np.sin(headingRad)
        pathX.append(currentX)
        pathY.append(currentY)
        
    elif eventType == "turn":
        #change direction
        angleChange = turnAngles[eventIdx]
        currentHeading += angleChange
        currentHeading = currentHeading % 360
        
        turnX.append(currentX)
        turnY.append(currentY)

pathX = np.array(pathX)
pathY = np.array(pathY)

#plot the walking simulation
fig = plt.figure(figsize=(12, 12))
ax = plt.gca()

#plot path
ax.plot(pathX, pathY, 'b-', linewidth=3, label='Walking Path', zorder=2)

#start and end points
ax.plot(pathX[0], pathY[0], "go", markersize=10, label="Start", markeredgecolor='black', markeredgewidth=2, zorder=5)
ax.plot(pathX[-1], pathY[-1], "rs", markersize=10, label='End', markeredgecolor='black', markeredgewidth=2, zorder=5)

#mark turns
for i in range(len(turnX)):
    ax.plot(turnX[i], turnY[i], "r^", markersize=8, markeredgecolor='black', markeredgewidth=1.5, zorder=4)

#add step markers
for i in range(0, len(pathX), 1):
    ax.plot(pathX[i], pathY[i], 'ko', markersize=5, alpha=0.5, zorder=3)

#formatting
ax.grid(True, alpha=0.4, linestyle='--', linewidth=1)
ax.set_aspect('equal')
ax.set_xlabel('East-West Distance (m)', fontsize=13, fontweight='bold')
ax.set_ylabel('North-South Distance (m)', fontsize=13, fontweight='bold')
ax.set_title(f'Walking Trajectory', fontsize=15, fontweight='bold', pad=20)
ax.legend(fontsize=12, loc='best', framealpha=0.95, edgecolor='black')

plt.show()