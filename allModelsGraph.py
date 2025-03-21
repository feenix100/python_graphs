import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Given data
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([20, 40, 75, 150, 297, 510])

# Define model functions
def linear_model(x, m, b):
    return m * x + b

def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c

def exponential_model(x, a, k):
    return a * np.exp(k * x)

def logarithmic_model(x, a, b):
    return a * np.log(x) + b

def power_model(x, a, b):
    return a * np.power(x, b)

# Fit the models
params_linear, _ = curve_fit(linear_model, x, y)
params_quadratic, _ = curve_fit(quadratic_model, x, y)
params_exponential, _ = curve_fit(exponential_model, x, y, p0=(20, 0.5))

# For logarithmic model, exclude x = 0 (since ln(0) is undefined)
x_log = x[1:]
y_log = y[1:]
params_logarithmic, _ = curve_fit(logarithmic_model, x_log, y_log, p0=(1, 20))

# For power model, exclude x = 0 (to avoid 0^b issues)
x_power = x[1:]
y_power = y[1:]
params_power, _ = curve_fit(power_model, x_power, y_power, p0=(20, 1))

# Generate fitted curves for plotting
x_fit = np.linspace(0, 5, 100)
y_fit_linear = linear_model(x_fit, *params_linear)
y_fit_quadratic = quadratic_model(x_fit, *params_quadratic)
y_fit_exponential = exponential_model(x_fit, *params_exponential)

# For logarithmic and power models, avoid x = 0:
x_fit_nonzero = np.linspace(0.1, 5, 100)
y_fit_logarithmic = logarithmic_model(x_fit_nonzero, *params_logarithmic)
y_fit_power = power_model(x_fit_nonzero, *params_power)

# Plot the data and the fitted models
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='black', label='Data')

plt.plot(x_fit, y_fit_linear, label=f'Linear: y = {params_linear[0]:.3f}x + {params_linear[1]:.3f}')
plt.plot(x_fit, y_fit_quadratic, label=f'Quadratic: y = {params_quadratic[0]:.3f}x² + {params_quadratic[1]:.3f}x + {params_quadratic[2]:.3f}')
plt.plot(x_fit, y_fit_exponential, label=f'Exponential: y = {params_exponential[0]:.3f}e^({params_exponential[1]:.3f}x)')
plt.plot(x_fit_nonzero, y_fit_logarithmic, label=f'Logarithmic: y = {params_logarithmic[0]:.3f}ln(x) + {params_logarithmic[1]:.3f}')
plt.plot(x_fit_nonzero, y_fit_power, label=f'Power: y = {params_power[0]:.3f}x^{params_power[1]:.3f}')

plt.xlabel('Time (hours)')
plt.ylabel('Bacteria Count')
plt.title('Bacteria Growth Data and Model Fits')
plt.legend()
plt.show()

# Function to calculate R^2
def calc_r2(y_true, y_pred):
    residuals = y_true - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res/ss_tot

# Compute R^2 for each model
r2_linear = calc_r2(y, linear_model(x, *params_linear))
r2_quadratic = calc_r2(y, quadratic_model(x, *params_quadratic))
r2_exponential = calc_r2(y, exponential_model(x, *params_exponential))
r2_logarithmic = calc_r2(y_log, logarithmic_model(x_log, *params_logarithmic))
r2_power = calc_r2(y_power, power_model(x_power, *params_power))

print("R^2 Values:")
print("Linear:", r2_linear)
print("Quadratic:", r2_quadratic)
print("Exponential:", r2_exponential)
print("Logarithmic:", r2_logarithmic)
print("Power:", r2_power)
