import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import data
data = pd.read_csv("ACCELERATION.csv")

#get each column of the data
timestamps = data['timestamp'].values
acceleration = data['acceleration'].values
noisyAcceleration = data['noisyacceleration'].values

#change in time for each observation
dt = 0.1

#create 0 populated array for velocity
velocity = np.zeros(len(acceleration))
velocityNoisy = np.zeros(len(noisyAcceleration))

#populate velocity array
for i in range(1, len(acceleration)):
    velocity[i] = velocity[i-1] + acceleration[i] * dt
    velocityNoisy[i] = velocityNoisy[i-1] + noisyAcceleration[i] * dt

#create 0 populated array for distance
distance = np.zeros(len(velocity))
distanceNoisy = np.zeros(len(velocityNoisy))

#populate distance array
for i in range(1, len(velocity)):
    distance[i] = distance[i-1] + velocity[i] * dt
    distanceNoisy[i] = distanceNoisy[i-1] + velocityNoisy[i] * dt

#create plots
fig, axes = plt.subplots(3, 1, figsize = (10, 12))

#plot 1 acceleration vs time
axes[0].plot(timestamps, acceleration, label="Actual", linewidth = 2)
axes[0].plot(timestamps, noisyAcceleration, label="Noisy", alpha = 0.7)
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Acceleration (m/s²)")
axes[0].set_title("Acceleration vs Time")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

#plot 2 velocity vs time
axes[1].plot(timestamps, velocity, label="Actual", linewidth=2)
axes[1].plot(timestamps, velocityNoisy, label="Noisy", alpha = 0.7)
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Velocity (m/s)")
axes[1].set_title("Velocity VS Time")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

#plot 3 distance vs time
axes[2].plot(timestamps, distance, label="Actual", linewidth=2)
axes[2].plot(timestamps, distanceNoisy, label="Noisy", alpha=0.7)
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Distance (m)")
axes[2].set_title("Distance vs Time")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#print results
print(f"Final distance using actual acceleration: {distance[-1]} m")
print(f"Final distance using noisy acceleration:  {distanceNoisy[-1]} m")
print(f"Difference in distance estimates:         {abs(distance[-1] - distanceNoisy[-1])} m")
