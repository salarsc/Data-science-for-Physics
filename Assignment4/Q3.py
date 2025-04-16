
"""
Created on Mon Apr 9 03:54:41 2025

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
ys = np.sin(omega0 * xs)/(omega0 * xs) - 0.15 * np.cos(omega1 * xs)
ys += yerrs * rng.normal(size=xs.shape)

n = len(xs)


def fourier_design_matrix(t, p, T=3.0):
    
    X = np.zeros((len(t), p))
    for j in range(p):
        freq = (np.floor((j+1)/2) * np.pi) / T
        if (j+1) % 2 == 1:
            X[:, j] = np.cos(freq * t)
        else:
            X[:, j] = np.sin(freq * t)
    return X


def fit_ols(xs, ys, p, T=3.0):
    
    X = fourier_design_matrix(xs, p, T=T)
    beta, _, _, _ = np.linalg.lstsq(X, ys, rcond=None)
    return beta


def fit_weighted_ols(xs, ys, p, f, T=3.0):
    
    
    X_raw = fourier_design_matrix(xs, p, T=T)
    
    
    freqs = []
    for j in range(p):
        freqs.append( (np.floor((j+1)/2) * np.pi) / T )
    
    
    X_scaled = np.zeros_like(X_raw)
    for j in range(p):
        wj = f(freqs[j])  # weight for this freq
        X_scaled[:, j] = X_raw[:, j] * wj
    
    
    beta_scaled, _, _, _ = np.linalg.lstsq(X_scaled, ys, rcond=None)
    
    
    beta = np.zeros_like(beta_scaled)
    for j in range(p):
        wj = f(freqs[j])
        beta[j] = beta_scaled[j] * wj
    
    return beta


def matern32_kernel(x1, x2, s=0.05):
    
    prefactor = np.sqrt(np.pi/8)
    r = np.abs(x1 - x2)
    return prefactor * (1.0 + r/s) * np.exp(-r/s)

def gp_predict_matern32(xs, ys, xstar, s=0.05):
   
    n = len(xs)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i,j] = matern32_kernel(xs[i], xs[j], s=s)
    
    alpha = np.linalg.solve(K, ys)
    
   
    ystar = np.zeros_like(xstar)
    for i, xx in enumerate(xstar):
        k_star = np.array([matern32_kernel(xx, xi, s=s) for xi in xs])
        ystar[i] = np.dot(k_star, alpha)
    
    return ystar


p = 1024            
s_param = 0.05      


beta_ols = fit_ols(xs, ys, p=p, T=3.0)


def f_weight(omega):
    return 1.0 / (s_param**2 * omega**2 + 1.0)

beta_wols = fit_weighted_ols(xs, ys, p=p, f=f_weight, T=3.0)


xgrid = np.linspace(0, 1, 500)
ygp   = gp_predict_matern32(xs, ys, xgrid, s=s_param)


Xg = fourier_design_matrix(xgrid, p, T=3.0)
y_ols = Xg @ beta_ols


y_wols = Xg @ beta_wols

plt.figure(figsize=(8,6))
plt.errorbar(xs, ys, yerrs, fmt='k.', label='Data', alpha=0.8, zorder=10)

plt.plot(xgrid, y_ols, 'b-', lw=2, label='OLS')
plt.plot(xgrid, y_wols, 'g-', lw=2, label='OLS with feature weights')
plt.plot(xgrid, ygp, 'r-', lw=2, label='GP (Matern 3/2)')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-0.4, 1.5)
plt.ylim(-0.4, 1.5)
plt.title("Comparison of OLS, Weighted OLS, and GP (Matérn 3/2)\nReplicating Figure 10 in Hogg & Villar")
plt.legend()
plt.tight_layout()
plt.savefig("Fig10_equivalent.png")
plt.show()
