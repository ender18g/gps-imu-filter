import numpy as np

import matplotlib.pyplot as plt
import math

# Define Kalman filter parameters
dt = 0.005  # Time step
gps_hz = 0.7  # GPS measurement frequency


gps_period = 1 / gps_hz
dt_per_gps = math.floor(gps_period / dt)

A = np.array([
    [1, 0, 0, dt, 0, 0],
    [0, 1, 0, 0, dt, 0],
    [0, 0, 1, 0, 0, dt],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1]
  ])  # State transition matrix

B = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, dt, 0, 0],
    [0, 0, 0, 0, dt, 0],
    [0, 0, 0, 0, 0, dt]
    ])  # Control matrix

H = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0]
  ])  # Measurement matrix


Q = np.eye(6) * 0.01  # Process noise covariance
R = np.eye(3) * 0.01  # Measurement noise covariance

# Generate simulated data
t = np.arange(0, 30, dt)
n = len(t)
# actual path is x increasing linearly, y and z increasing sinusoidally
actual_path = np.array([t, np.sin(t), 0.2 * np.cos(t)])
actual_velocity = np.array([np.ones(n), np.cos(t), -0.2 * np.sin(t)])
actual_acceleration = np.array([np.zeros(n), -np.sin(t), -0.2 * np.cos(t)])
# gps measured path has noise
gps_path = actual_path + np.random.randn(3, n) * 0.01
# accel has noise
accelerometer_data = actual_acceleration + np.random.randn(3, n) * 0.05 + np.array([0,0.001,0]).reshape((3, 1)) #accelerometer has bias
gps_measured_path = np.empty((3, 0))
accel_measured_path = np.empty((3, 0))


# Kalman filter initialization
x0 = np.array([actual_path[:, 0], actual_velocity[:, 0]]).flatten().reshape((6, 1))  # Initial state

P = np.eye(6) * 0.1  # Initial state covariance

x = x0  # Initial state estimate
# integrate accel data to make IMU path
for i in range(n):
  u = np.array([0, 0, 0, accelerometer_data[0, i], accelerometer_data[1, i], accelerometer_data[2, i]]).reshape((6, 1))
  x = np.dot(A, x) + np.dot(B, u)
  accel_measured_path = np.append(accel_measured_path, x[:3], axis=1)


x = x0  # Initial state estimate
# Kalman filter loop
kalman_path = np.zeros((3, n))
for i in range(n):
  u = np.array([0, 0, 0, accelerometer_data[0, i], accelerometer_data[1, i], accelerometer_data[2, i]]).reshape((6, 1))
  # Update state estimate and covariance
  x = np.dot(A, x) + np.dot(B, u)
  P = np.dot(np.dot(A, P), A.T) + Q

  
  # Kalman gain
  K = np.dot(np.dot(P, H.T), np.linalg.inv(np.dot(np.dot(H, P), H.T) + R))
  
  # GPS is measured less frequently
  if i % dt_per_gps == 0:
    # Update state estimate and covariance
    z = gps_path[:, i].reshape((3, 1))
    x = x + np.dot(K, (z - np.dot(H, x)))
    P = np.dot((np.eye(6) - np.dot(K, H)), P)
    # append to the gps_measured_path
    gps_measured_path = np.append(gps_measured_path, x[:3], axis=1)
    

  
  # Save Kalman path
  kalman_path[:, i] = x[:3].flatten()
  
# Plot paths
plt.figure()
plt.grid()
options = {'markersize': 8, 'linewidth': 2, 'alpha': 0.6}
plt.plot(actual_path[0], actual_path[1], label='Actual Path', **options)
plt.plot(gps_measured_path[0], gps_measured_path[1], label='GPS Measured Path', **options)
plt.plot(kalman_path[0], kalman_path[1], label='Kalman Filtered Path', **options, linestyle='--')
plt.plot(accel_measured_path[0], accel_measured_path[1], label='IMU Integrated Path', **options)
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Path')

# make figure bigger
fig = plt.gcf()
fig.set_size_inches(12, 8)

# save the plot
plt.savefig('kalman_filter.png')