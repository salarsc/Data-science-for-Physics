"""


@author: Salar Ghaderi
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt


with open('timeseries.pkl', 'rb') as f:
    times, ys, ivars = pickle.load(f)


sidereal_year = 365.256363004  # days
sidereal_day = sidereal_year / (sidereal_year + 1.)
omega = 2 * np.pi / sidereal_day

N = len(times)  # number of data points

def fit_model(times, y, ivars, k):
    
    X_columns = [np.ones_like(times)]
    
    for j in range(1, k+1):
        X_columns.append(np.cos(j * omega * times))
        X_columns.append(np.sin(j * omega * times))
    X = np.column_stack(X_columns)
    
    
    W = np.sqrt(ivars)
    Xw = X * W[:, None]
    yw = y * W
    
    
    p, residuals, rank, s = np.linalg.lstsq(Xw, yw, rcond=None)
    model = X @ p
    chi2_val = np.sum(ivars * (y - model)**2)
    return p, chi2_val, model

def compute_BIC(chi2, num_params, N):
    
    return chi2 + num_params * np.log(N)

max_k = 3  # Maximum number of harmonics to consider
results = {}

print("Comparing models using BIC:")
for i, y in enumerate(ys):
    print(f"\nLight curve {i}:")
    bic_values = []
    chi2_values = []
    
    
    for k in range(0, max_k+1):
        p, chi2_val, model = fit_model(times, y, ivars, k)
        num_params = 1 + 2 * k  # 1 parameter for mu, 2 for each harmonic
        bic = compute_BIC(chi2_val, num_params, N)
        bic_values.append(bic)
        chi2_values.append(chi2_val)
        print(f"  k = {k:1d} harmonics: chi2 = {chi2_val:7.2f}, num_params = {num_params:2d}, BIC = {bic:7.2f}")
    
    best_k = np.argmin(bic_values)
    print(f"  --> Best model order: {best_k} harmonic(s) (lowest BIC = {bic_values[best_k]:.2f})")
    results[i] = {"bic": bic_values, "chi2": chi2_values, "best_k": best_k}
