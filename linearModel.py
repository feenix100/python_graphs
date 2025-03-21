import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Given data: Time in hours and Bacteria Count
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([20, 40, 75, 150, 297, 510])

# Define the linear model function: y = m*x + b
def linear_model(x, m, b):
    return m * x + b

# Fit the linear model to the data using curve_fit
params, covariance = curve_fit(linear_model, x, y)
m, b = params  # m is the slope, b is the y-intercept

# Generate a range of x-values for plotting the fitted line
x_fit = np.linspace(np.min(x), np.max(x), 100)
y_fit = linear_model(x_fit, m, b)

# Create a scatter plot of the data and overlay the linear fit
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label='Data')
plt.plot(x_fit, y_fit, color='red', label=f'Linear Fit: y = {m:.3f}x + {b:.3f}')
plt.xlabel('Time (hours)')
plt.ylabel('Bacteria Count')
plt.title('Bacterial Growth: Linear Model')
plt.legend()
plt.grid(True)
plt.show()
