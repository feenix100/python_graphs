import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Given data: Time (hours) and Bacteria Count
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([20, 40, 75, 150, 297, 510])

# Define the exponential model function: y = a * e^(k*x)
def exponential_model(x, a, k):
    return a * np.exp(k * x)

# Fit the exponential model to the data using curve_fit
# Initial guess p0 can be provided to help the algorithm converge
params, covariance = curve_fit(exponential_model, x, y, p0=(20, 0.5))
a, k = params  # Extract coefficients a and k

# Generate a range of x-values for plotting the fitted curve
x_fit = np.linspace(np.min(x), np.max(x), 100)
y_fit = exponential_model(x_fit, a, k)

# Create a scatter plot of the data and overlay the exponential fit
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label='Data')
plt.plot(x_fit, y_fit, color='red', 
         label=f'Exponential Fit: y = {a:.3f}e^({k:.3f}x)')
plt.xlabel('Time (hours)')
plt.ylabel('Bacteria Count')
plt.title('Bacterial Growth: Exponential Model')
plt.legend()
plt.grid(True)
plt.show()
