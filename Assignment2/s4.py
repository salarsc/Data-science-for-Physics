# -*- coding: utf-8 -*-
"""


@author: Salar Ghaderi
"""

import numpy as np
import matplotlib.pyplot as plt

#helper func

def wls_fit(x, y, sy):
    """
    Perform weighted least-squares fit to the linear model y = b + m*x.
    Returns:
        m_best, b_best, var_m, var_b, cov_mb
    """
    # Construct design matrix: each row is [1, x_i]
    A = np.column_stack((np.ones_like(x), x))  # shape (N,2)
    # Inverse covariance:
    invC = np.diag(1.0 / sy**2)  # shape (N,N)

    # Solve (A^T C^-1 A)^-1 (A^T C^-1 Y)
    ATinvC = A.T @ invC
    cov_params = np.linalg.inv(ATinvC @ A)  # 2x2 matrix
    X_best = cov_params @ (ATinvC @ y)

    b_best, m_best = X_best
    # Extract parameter covariances
    var_b = cov_params[0,0]
    cov_mb = cov_params[0,1]
    var_m = cov_params[1,1]
    return m_best, b_best, var_m, var_b, cov_mb


def jackknife_slope(x, y, sy):
    """
    Return the jackknife estimate of slope variance and the array of slope fits.
    """
    N = len(x)
    m_list = []

    for i in range(N):
        # Exclude index i
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        x_j = x[mask]
        y_j = y[mask]
        sy_j = sy[mask]

        m_i, b_i, vm_i, vb_i, cmb_i = wls_fit(x_j, y_j, sy_j)
        m_list.append(m_i)

    m_list = np.array(m_list)
    m_mean = np.mean(m_list)

    # Jackknife variance:
    # (N-1)/N * sum( (m_i - m_mean)^2 )
    var_m_jack = (N - 1) / N * np.sum((m_list - m_mean)**2)
    return var_m_jack, m_list


def bootstrap_slope(x, y, sy, nboot=2000, rng=None):
    """
    Return the bootstrap estimate of slope variance, plus the array of slopes.
    """
    if rng is None:
        rng = np.random.default_rng()

    N = len(x)
    m_boot = []

    for _ in range(nboot):
        # Sample with replacement
        indices = rng.integers(low=0, high=N, size=N)
        x_b = x[indices]
        y_b = y[indices]
        sy_b = sy[indices]

        m_b, b_b, vm_b, vb_b, cmb_b = wls_fit(x_b, y_b, sy_b)
        m_boot.append(m_b)

    m_boot = np.array(m_boot)
    var_m_boot = np.var(m_boot, ddof=1)  # sample variance, unbiased
    return var_m_boot, m_boot


def plot_and_save(x, y, sy, m_best, b_best, m_boot, fname="fit.png", title=""):
    
    plt.figure(figsize=(6,5))
    plt.errorbar(x, y, yerr=sy, fmt='k.', label="Data", alpha=0.7)

    
    xx = np.linspace(x.min(), x.max(), 300)
    yy_best = b_best + m_best * xx
    plt.plot(xx, yy_best, 'r-', label=f"Best-fit slope = {m_best:.3f}")

    
    rng = np.random.default_rng(42)
    nlines = 12
    
    idx_samples = rng.choice(len(m_boot), size=nlines, replace=False)
    for idx in idx_samples:
        m_i = m_boot[idx]
        # We keep b = b_best, so lines pivot around that intercept:
        y_boot_line = b_best + m_i * xx
        plt.plot(xx, y_boot_line, color='gray', alpha=0.3)

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Plot saved to: {fname}")



#  MAIN SCRIPT


def main():
    
    x_ex2  = np.array([ 58,  98, 131, 146, 157, 158, 160, 165, 186, 198, 201, 202, 203, 210, 218, 244, 287])
    y_ex2  = np.array([173, 510, 311, 344, 317, 416, 337, 393, 423, 510, 442, 504, 495, 479, 533, 401, 402])
    sy_ex2 = np.array([ 15,  30,  16,  22,  52,  16,  31,  14,  42,  30,  25,  14,  21,  27,  16,  25,  15])

    x_ex1  = np.array([ 58, 146, 157, 158, 160, 165, 186, 198, 201, 202, 203, 210])
    y_ex1  = np.array([173, 344, 317, 416, 337, 393, 423, 510, 442, 504, 495, 479])
    sy_ex1 = np.array([ 15,  22,  52,  16,  31,  14,  42,  30,  25,  14,  21,  27])

    print("=== EXERCISE 2 DATA ===")
    
    m2, b2, vm2, vb2, cmb2 = wls_fit(x_ex2, y_ex2, sy_ex2)
    sm2_matrix = np.sqrt(vm2)

    #  Jackknife
    var_m2_jack, m_jack_ex2 = jackknife_slope(x_ex2, y_ex2, sy_ex2)
    sm2_jack = np.sqrt(var_m2_jack)

    #  Bootstrap
    var_m2_boot, m_boot_ex2 = bootstrap_slope(x_ex2, y_ex2, sy_ex2, nboot=2000)
    sm2_boot = np.sqrt(var_m2_boot)

    print(f"  Standard Fit: slope m = {m2:.3f} ± {sm2_matrix:.3f}, intercept b = {b2:.3f}")
    print(f"  Jackknife slope uncertainty: {sm2_jack:.3f}")
    print(f"  Bootstrap slope uncertainty: {sm2_boot:.3f}")
    
    plot_and_save(
        x_ex2, y_ex2, sy_ex2,
        m2, b2,
        m_boot_ex2,
        fname="ex2_fit.png",
        title="Exercise 2 Data"
    )


    print("\n=== EXERCISE 1 DATA ===")
    # 1) Standard Fit
    m1, b1, vm1, vb1, cmb1 = wls_fit(x_ex1, y_ex1, sy_ex1)
    sm1_matrix = np.sqrt(vm1)

    # 2) Jackknife
    var_m1_jack, m_jack_ex1 = jackknife_slope(x_ex1, y_ex1, sy_ex1)
    sm1_jack = np.sqrt(var_m1_jack)

    # 3) Bootstrap
    var_m1_boot, m_boot_ex1 = bootstrap_slope(x_ex1, y_ex1, sy_ex1, nboot=2000)
    sm1_boot = np.sqrt(var_m1_boot)

    print(f"  Standard Fit: slope m = {m1:.3f} ± {sm1_matrix:.3f}, intercept b = {b1:.3f}")
    print(f"  Jackknife slope uncertainty: {sm1_jack:.3f}")
    print(f"  Bootstrap slope uncertainty: {sm1_boot:.3f}")
    # 4) Plot & Save
    plot_and_save(
        x_ex1, y_ex1, sy_ex1,
        m1, b1,
        m_boot_ex1,
        fname="ex1_fit.png",
        title="Exercise 1 Data"
    )


if __name__ == "__main__":
    main()
