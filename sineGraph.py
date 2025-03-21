import numpy as np
import matplotlib.pyplot as plt

# Define the function f(x) = 2 + 3 sin((pi/2) x)
def f(x):
    return 2 + 3 * np.sin((np.pi/2) * x)

# Create an array of x-values. Since the period of the function is T = 2π/(π/2) = 4,
# we choose a range that covers a few periods. Here, we use x from -8 to 8.
x = np.linspace(-8, 8, 1000)

# Compute f(x) for each x-value
y = f(x)

# Create the plot
plt.figure(figsize=(10, 5))
plt.plot(x, y, label=r'$f(x)=2+3\sin\left(\frac{\pi}{2}x\right)$', color='blue')

# Plot the midline y=2
plt.axhline(2, color='red', linestyle='--', label='Midline: y = 2')

# Add labels, title, and legend
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Graph of f(x)=2+3 sin((π/2)x) with Midline')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()