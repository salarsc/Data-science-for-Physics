# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 22:26:27 2025

@author: Salar
"""

import numpy as np
import matplotlib.pyplot as plt




table_data = np.array([
    [ 1, 201, 592, 61,  9, -0.84],
    [ 2, 244, 401, 25,  4,  0.31],
    [ 3,  47, 583, 38, 11,  0.64],
    [ 4, 287, 402, 15,  7, -0.27],
    [ 5, 203, 495, 21,  5, -0.33],
    [ 6,  58, 173, 15,  9,  0.67],
    [ 7, 210, 479, 27,  4, -0.02],
    [ 8, 202, 504, 14,  4, -0.05],
    [ 9, 198, 510, 30, 11, -0.84],
    [10, 158, 416, 16,  7, -0.69],
    [11, 165, 393, 14,  5,  0.30],
    [12, 201, 442, 25,  5, -0.46],
    [13, 157, 317, 52,  5, -0.03],
    [14, 131, 311, 16,  6,  0.50],
    [15, 166, 400, 34,  6,  0.73],
    [16, 160, 337, 31,  5, -0.52],
    [17, 186, 423, 42,  9,  0.90],
    [18, 125, 334, 26,  8,  0.40],
    [19, 218, 533, 16,  6, -0.78],
    [20, 146, 344, 22,  5, -0.56]
])


x_vals = table_data[:, 1]
y_vals = table_data[:, 2]
sigma_y = table_data[:, 3]
N = len(x_vals)

# Weighted least-squares: y = b + m x

A = np.column_stack((np.ones(N), x_vals))
Y = y_vals
C = np.diag(sigma_y**2)
C_inv = np.linalg.inv(C)

# Solve for [b, m] and get covariance
M = A.T @ C_inv @ A
cov_params = np.linalg.inv(M)
X_best = cov_params @ (A.T @ C_inv @ Y)
b_best, m_best = X_best

# Extract slope variance
var_b = cov_params[0, 0]
var_m = cov_params[1, 1]
sigma_b = np.sqrt(var_b)
sigma_m = np.sqrt(var_m)

#plot
plt.figure(figsize=(8, 5), dpi=130)

plt.errorbar(
    x_vals, y_vals, yerr=sigma_y,
    fmt='o', markersize=6, markerfacecolor='white',
    markeredgecolor='navy', ecolor='skyblue',
    capsize=3, elinewidth=1,
    label='All Data (ID=1..20)'
)

x_fit = np.linspace(min(x_vals)-10, max(x_vals)+10, 200)
y_fit = b_best + m_best*x_fit
plt.plot(
    x_fit, y_fit, color='darkred', linewidth=2,
    label='Best-fit line'
)

line_eq = (
    f"y = ({b_best:.2f} ± {sigma_b:.2f}) + "
    f"({m_best:.2f} ± {sigma_m:.2f}) x"
)
extras = f"var(m) = {var_m:.4f} (slope variance)"
annot_str = line_eq + "\n" + extras

plt.text(
    0.03, 0.90, annot_str,
    transform=plt.gca().transAxes, va='top', fontsize=10,
    bbox=dict(facecolor='whitesmoke', edgecolor='gray', alpha=0.7)
)

plt.title("Exercise 2: Weighted Least-Squares with All Points (ID=1..20)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.savefig("Ex2_all_points.png", dpi=150)
plt.show()


print("==== Weighted Least-Squares Fit (All 20 Data Points) ====")
print(f"Intercept b = {b_best:.2f} ± {sigma_b:.2f}")
print(f"Slope m     = {m_best:.2f} ± {sigma_m:.2f}")
print(f"var(m)      = {var_m:.4f}")
print()

