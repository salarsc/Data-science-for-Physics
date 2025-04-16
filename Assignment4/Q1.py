"""
Created on Sun Apr 9 17:54:46 2025

@author: Salar
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# Generation of Data

rng = np.random.default_rng(17)
small = 0.05
xs = np.arange(0.0 + 0.5 * small, 1.0, small)
xs += 0.05 * small * rng.normal(size=xs.shape)
yerrs = 0.02 + 0.06 * rng.uniform(size=xs.shape)
omega0 = 5.0
omega1 = np.pi ** 3
ys = np.sin(omega0 * xs) / (omega0 * xs) - 0.15 * np.cos(omega1 * xs)
ys += yerrs * rng.normal(size=xs.shape)


xgrid = np.linspace(0, 1, 500)


f_linear = interp1d(xs, ys, kind='linear', fill_value="extrapolate")
y_linear = f_linear(xgrid)

plt.figure(figsize=(8, 5))
plt.errorbar(xs, ys, yerr=yerrs, fmt="k.", label="Data points")
plt.plot(xgrid, y_linear, 'r-', label="Linear interp")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Interpolation")
plt.legend()
plt.tight_layout()
plt.savefig("linear_interp.png")
plt.show()


f_cubic = interp1d(xs, ys, kind='cubic', fill_value="extrapolate")
y_cubic = f_cubic(xgrid)

plt.figure(figsize=(8, 5))
plt.errorbar(xs, ys, yerr=yerrs, fmt="k.", label="Data points")
plt.plot(xgrid, y_cubic, 'b-', label="Cubic spline interp")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Cubic Spline Interpolation")
plt.legend()
plt.tight_layout()
plt.savefig("cubic_interp.png")  

def lanczos_kernel(z, a):
    
    out = np.sinc(z) * np.sinc(z / a)
    out[np.abs(z) >= a] = 0.0
    return out

def lanczos_interp(xgrid, xs, ys, a=5):
    
    T = np.median(np.diff(np.sort(xs)))
    y_interp = np.zeros_like(xgrid)
    for j, x_val in enumerate(xgrid):
        distances = (x_val - xs) / T
        mask = np.abs(distances) < a  
        if np.any(mask):
            weights = lanczos_kernel(distances[mask], a)
            y_interp[j] = np.sum(weights * ys[mask]) / np.sum(weights)
        else:
            y_interp[j] = np.nan  
    return y_interp


y_lanczos = lanczos_interp(xgrid, xs, ys, a=5)

plt.figure(figsize=(8, 5))
plt.errorbar(xs, ys, yerr=yerrs, fmt="k.", label="Data points")
plt.plot(xgrid, y_lanczos, 'g-', label="Lanczos (5 taps) interp")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Lanczos 5-Tap Interpolation")
plt.legend()
plt.tight_layout()
plt.savefig("lanczos_interp.png")  

