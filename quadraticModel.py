import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Given data: Time (hours) and Bacteria Count
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([20, 40, 75, 150, 297, 510])

# Define the quadratic model function: y = ax^2 + bx + c
def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c

# Fit the quadratic model to the data using curve_fit
params, covariance = curve_fit(quadratic_model, x, y)
a, b, c = params  # Extract coefficients a, b, and c

# Generate a range of x-values for plotting the fitted curve
x_fit = np.linspace(np.min(x), np.max(x), 100)
y_fit = quadratic_model(x_fit, a, b, c)

# Create a scatter plot of the data and overlay the quadratic fit
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label='Data')
plt.plot(x_fit, y_fit, color='red', label=f'Quadratic Fit: y = {a:.3f}x² + {b:.3f}x + {c:.3f}')
plt.xlabel('Time (hours)')
plt.ylabel('Bacteria Count')
plt.title('Bacterial Growth: Quadratic Model')
plt.legend()
plt.grid(True)
plt.show()
