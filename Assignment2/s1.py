"""
@author: Salar Ghaderi
"""

import pickle
import numpy as np
from scipy.stats import chi2


pkl_filename = "timeseries.pkl"

# data
with open(pkl_filename, "rb") as f:
    times, y_block, ivar = pickle.load(f)


# The number of time points:
n_points = len(times)
# Degrees of freedom = number of data points - number of free parameters (1 mean)
dof = n_points - 1

# best-fit constant model and chi-square
p_values = []
for i in range(y_block.shape[0]):
    y_i = y_block[i]      # The i-th light curve
    w_i = ivar            # The corresponding weights (inverse variance)
    
    # Weighted mean (best-fit constant)
    mean_i = np.sum(y_i * w_i) / np.sum(w_i)
    
    # chi-square
    chi2_val = np.sum((y_i - mean_i)**2 * w_i)
    
    # p-value = 1 - CDF(chi2) for dof = 1023
    p_val = 1.0 - chi2.cdf(chi2_val, dof)
    p_values.append(p_val)
    
    print(f"Light Curve {i}: χ² = {chi2_val:.2f}, p-value = {p_val:.5e}")


min_pval_index = np.argmin(p_values)
max_pval_index = np.argmax(p_values)


