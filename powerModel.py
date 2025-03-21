import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Given data: Time (hours) and Bacteria Count
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([20, 40, 75, 150, 297, 510])

# Exclude x = 0 for the power model
x_power = x[1:]
y_power = y[1:]

# Define the power model function: y = a * x^b
def power_model(x, a, b):
    return a * np.power(x, b)

# Fit the power model to the data using curve_fit
params, covariance = curve_fit(power_model, x_power, y_power, p0=(20, 1))
a, b = params  # Extract the coefficients a and b

# Generate a range of x-values for plotting (x > 0)
x_fit = np.linspace(0.1, 5, 100)
y_fit = power_model(x_fit, a, b)

# Create a scatter plot of the data (for x > 0) and overlay the power model fit
plt.figure(figsize=(8, 6))
plt.scatter(x_power, y_power, color='blue', label='Data (x > 0)')
plt.plot(x_fit, y_fit, color='red', label=f'Power Fit: y = {a:.3f} * x^{b:.3f}')
plt.xlabel('Time (hours)')
plt.ylabel('Bacteria Count')
plt.title('Bacterial Growth: Power Model')
plt.legend()
plt.grid(True)
plt.show()
