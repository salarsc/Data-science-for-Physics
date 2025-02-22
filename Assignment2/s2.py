
"""
Created on Fri Feb 21 18:31:58 2025

@author: Salar Ghaderi
"""

import numpy as np
import pickle
from scipy.stats import chi2

 
with open('timeseries.pkl', 'rb') as f:
    times, ys, ivars = pickle.load(f)


sidereal_year = 365.256363004
sidereal_day = sidereal_year / (sidereal_year + 1.)
SIDEREAL_OMEGA = 2 * np.pi / sidereal_day

def chisq_null(y, ivars):
    """Compute chi-square for the null (constant) model."""
    
    mean = np.sum(ivars * y) / np.sum(ivars)
    return np.sum(ivars * (y - mean)**2)

def fit_harmonic(times, y, ivars, omega):
    """
    Fit a model that is a constant plus a harmonic signal at frequency omega.
    This model is linear if written as:
        f(t) = a0 + a1*cos(omega*t) + a2*sin(omega*t).
    """
    
    X = np.column_stack((np.ones_like(times), np.cos(omega * times), np.sin(omega * times)))
    
    XT_W = X.T * ivars  # each row of X.T multiplied by ivars
    A = XT_W @ X       # (3x3 matrix)
    b = XT_W @ y       # (3-vector)
    p = np.linalg.solve(A, b)
    model = X @ p
    chi2_val = np.sum(ivars * (y - model)**2)
    return p, chi2_val

print("Comparing null model to constant-plus-harmonic model:")
for i, y in enumerate(ys):
    
    chi2_null_val = chisq_null(y, ivars)
    
    p, chi2_harm = fit_harmonic(times, y, ivars, SIDEREAL_OMEGA)
    
    delta_logL = 0.5 * (chi2_null_val - chi2_harm)
    
    amplitude = np.sqrt(p[1]**2 + p[2]**2)
    print(f"Light curve {i}:")
    print(f"  Null chi2       = {chi2_null_val:.2f}")
    print(f"  Harmonic chi2   = {chi2_harm:.2f}")
    print(f"  Delta logL      = {delta_logL:.2f}")
    print(f"  Best-fit amplitude = {amplitude:.5f}")
    print("")
