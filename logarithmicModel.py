import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Given data: Time (hours) and Bacteria Count
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([20, 40, 75, 150, 297, 510])

# Exclude x = 0 for logarithmic fitting (ln(0) is undefined)
x_log = x[1:]
y_log = y[1:]

# Define the logarithmic model function: y = a * ln(x) + b
def logarithmic_model(x, a, b):
    return a * np.log(x) + b

# Fit the logarithmic model to the data using curve_fit
params, covariance = curve_fit(logarithmic_model, x_log, y_log, p0=(1, 20))
a, b = params  # Extract coefficients a and b

# Generate a range of x-values for plotting (x > 0)
x_fit = np.linspace(0.1, np.max(x), 100)
y_fit = logarithmic_model(x_fit, a, b)

# Create a scatter plot of the data (for x > 0) and overlay the logarithmic fit
plt.figure(figsize=(8, 6))
plt.scatter(x_log, y_log, color='blue', label='Data (x > 0)')
plt.plot(x_fit, y_fit, color='red', label=f'Logarithmic Fit: y = {a:.3f}ln(x) + {b:.3f}')
plt.xlabel('Time (hours)')
plt.ylabel('Bacteria Count')
plt.title('Bacterial Growth: Logarithmic Model')
plt.legend()
plt.grid(True)
plt.show()
