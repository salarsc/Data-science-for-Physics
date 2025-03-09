"""
Created on Fri Mar  7 

@author: Salar
"""
import numpy as np
import pickle
from sklearn.model_selection import KFold


with open('timeseries.pkl', 'rb') as f:
    times, ys, ivars = pickle.load(f)


sidereal_year = 365.256363004  # days
sidereal_day = sidereal_year / (sidereal_year + 1.)
omega = 2 * np.pi / sidereal_day  # base angular frequency

N = len(times)  # number of data points

def build_design_matrix(times, k, omega):
   
    X_cols = [np.ones_like(times)]
    for j in range(1, k+1):
        X_cols.append(np.cos(j * omega * times))
        X_cols.append(np.sin(j * omega * times))
    X = np.column_stack(X_cols)
    return X

def fit_weighted_ls(X, y, ivars):
    
    W = np.sqrt(ivars)
    Xw = X * W[:, None]
    yw = y * W
    
    p, residuals, rank, s = np.linalg.lstsq(Xw, yw, rcond=None)
    return p

def cv_log_likelihood(times, y, ivars, k, n_splits=5, random_state=42):
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    total_cv_logL = 0.0

    # Loop over CV folds
    for train_idx, val_idx in kf.split(times):
        # Training data
        times_train = times[train_idx]
        y_train = y[train_idx]
        ivars_train = ivars[train_idx]

        # Build design matrix for training set and fit model
        X_train = build_design_matrix(times_train, k, omega)
        p = fit_weighted_ls(X_train, y_train, ivars_train)
        
        # Validation data
        times_val = times[val_idx]
        y_val = y[val_idx]
        ivars_val = ivars[val_idx]
        X_val = build_design_matrix(times_val, k, omega)
        y_pred = X_val @ p
        
        # Compute log likelihood on validation set
        # For Gaussian errors: log L ~ -0.5 * sum(ivars*(y - y_pred)^2) (up to a constant)
        logL_val = -0.5 * np.sum(ivars_val * (y_val - y_pred)**2)
        total_cv_logL += logL_val

    return total_cv_logL


max_k = 3  
cv_results = {}

print("Cross-validation likelihood for each light curve:")
for i, y in enumerate(ys):
    print(f"\nLight curve {i}:")
    cv_logL_values = []
    for k in range(0, max_k+1):
        # Compute total CV log likelihood for model with k harmonics
        total_cv_logL = cv_log_likelihood(times, y, ivars, k, n_splits=5, random_state=42)
        cv_logL_values.append(total_cv_logL)
        num_params = 1 + 2 * k  # mu + 2 parameters per harmonic
        print(f"  k = {k:1d} harmonics |(params = {num_params:2d})| Total CV logL = {total_cv_logL:10.2f}")
    
    
    best_k = np.argmax(cv_logL_values)
    print(f"  Best model order: {best_k}harmonic(s)|(CV logL = {cv_logL_values[best_k]:.2f})")
    cv_results[i] = {"cv_logL":cv_logL_values, "best_k": best_k}
