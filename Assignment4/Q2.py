"""
Created on Sun Apr 9 17:54:46 2025

@author: Salar
"""
import numpy as np
import matplotlib.pyplot as plt


rng = np.random.default_rng(17)
small = 0.05
xs = np.arange(0.0 + 0.5 * small, 1.0, small)
xs += 0.05 * small * rng.normal(size=xs.shape)
yerrs = 0.02 + 0.06 * rng.uniform(size=xs.shape)
omega0 = 5.0
omega1 = np.pi ** 3
ys = np.sin(omega0 * xs) / (omega0 * xs) - 0.15 * np.cos(omega1 * xs)
ys += yerrs * rng.normal(size=xs.shape)

n = len(xs)


def fourier_design_matrix(t, p, T=3.0):
    
    X = np.zeros((len(t), p))
    for j in range(p):
        freq = (np.floor((j+1) / 2) * np.pi) / T
        if (j+1) % 2 == 1:
            X[:, j] = np.cos(freq * t)
        else:
            X[:, j] = np.sin(freq * t)
    return X


def fit_ols(xs, ys, p, T=3.0):
    X = fourier_design_matrix(xs, p, T=T)
    beta, _, _, _ = np.linalg.lstsq(X, ys, rcond=None)
    return beta


xgrid = np.linspace(0.0, 1.0, 500)


p_vals_under = [3, 7, 21]
plt.figure(figsize=(8, 6))
plt.errorbar(xs, ys, yerrs, fmt='k.', label='Data', alpha=0.8, zorder=10)

colors = ['r', 'g', 'b']
for color, p in zip(colors, p_vals_under):
    beta = fit_ols(xs, ys, p, T=3.0)
    Xg = fourier_design_matrix(xgrid, p, T=3.0)
    yg = Xg @ beta
    plt.plot(xgrid, yg, color=color, lw=2, label=f"OLS, p={p}")

plt.title("OLS fits with p < n (Figure 2 style)")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(-0.4, 1.5)
plt.ylim(-0.4, 1.5)
plt.legend()
plt.tight_layout()
plt.savefig("OLS_Fits_Under.png")  # Save the under-parameterized plot
plt.show()


p_vals_over = [30, 73, 2049]
plt.figure(figsize=(8, 6))
plt.errorbar(xs, ys, yerrs, fmt='k.', label='Data', alpha=0.8, zorder=10)

colors = ['r', 'g', 'b']
for color, p in zip(colors, p_vals_over):
    beta = fit_ols(xs, ys, p, T=3.0)
    Xg = fourier_design_matrix(xgrid, p, T=3.0)
    yg = Xg @ beta
    plt.plot(xgrid, yg, color=color, lw=2, label=f"OLS, p={p}")

plt.title("OLS fits with p > n (Figure 5 style)")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(-0.4, 1.5)
plt.ylim(-0.4, 1.5)
plt.legend()
plt.tight_layout()
plt.savefig("OLS_Fits_Over.png")  # Save the over-parameterized plot
plt.show()
