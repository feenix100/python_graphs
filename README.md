
# 📊 Simple Data Grapher

A collection of **Python scripts** that generate graphs from table data using common mathematical models. Each script includes hardcoded data and is ready to run for quick visualization and analysis.

### Supported Models:
- Linear
- Exponential
- Logarithmic
- Quadratic
- Sine
- Power

---

## 🛠️ Requirements

- Python 3.x
- NumPy
- Matplotlib
- SciPy

Install dependencies:
```bash
pip install numpy matplotlib scipy
```

---

## 🚀 Usage

1. **Open a script** for the model you want to use (e.g., `linear_model.py`).
2. **Adjust the data** inside the script in the `x_data` and `y_data` lists.
3. Run the script:
```bash
python linear_model.py
```

The script will:
- Fit the data to the chosen model.
- Plot the data points and best-fit curve.
- Display the graph and print the equation and R² value.

---

## 🗂 Example Script Structure

```python
# linear_model.py

import numpy as np
import matplotlib.pyplot as plt

# Sample Data (adjust as needed)
x_data = [1, 2, 3, 4, 5]
y_data = [2, 4.1, 6.2, 8.1, 10.3]

# Fit a linear model y = mx + b
coeffs = np.polyfit(x_data, y_data, 1)
m, b = coeffs
y_fit = np.polyval(coeffs, x_data)

# Plot
plt.scatter(x_data, y_data, label='Data')
plt.plot(x_data, y_fit, label=f'y = {m:.2f}x + {b:.2f}', color='red')
plt.legend()
plt.title('Linear Model Fit')
plt.show()
```

---

## 📚 Scripts in This Repo

| File Name            | Model Type     |
|----------------------|----------------|
| `linear_model.py`    | Linear         |
| `exponential_model.py` | Exponential  |
| `logarithmic_model.py` | Logarithmic  |
| `quadratic_model.py` | Quadratic      |
| `sine_model.py`      | Sine           |
| `power_model.py`     | Power          |

---

## 📝 Notes

- These are **simple scripts** with no file input/output – just adjust the data inside the script.
- Ideal for quick analysis, learning, and visual exploration of data trends.

---

## 📄 License

MIT License

---

Let me know if you'd like to include a short description for each model script or need help writing one of the example scripts!
