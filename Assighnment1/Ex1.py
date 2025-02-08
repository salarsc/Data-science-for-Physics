import numpy as np
import matplotlib.pyplot as plt


# Table data
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

# Selecting ID=5..20
subset = table_data[4:20]
x_vals = subset[:, 1]
y_vals = subset[:, 2]
sigma_y = subset[:, 3]
N = len(x_vals)

# Weighted least-squares
A = np.column_stack((np.ones(N), x_vals))
C = np.diag(sigma_y**2)
C_inv = np.linalg.inv(C)

M = A.T @ C_inv @ A
cov_params = np.linalg.inv(M)
X_best = cov_params @ (A.T @ C_inv @ y_vals)
b_best, m_best = X_best

var_b = cov_params[0, 0]
var_m = cov_params[1, 1]
sigma_b = np.sqrt(var_b)
sigma_m = np.sqrt(var_m)

# Plot
plt.figure(figsize=(8,5), dpi=130)
plt.errorbar(
    x_vals, y_vals, yerr=sigma_y,
    fmt='o', markersize=6, 
    markerfacecolor='white', markeredgecolor='navy',
    ecolor='skyblue', capsize=3, elinewidth=1,
    label='Data (ID=5..20)'
)

x_fit = np.linspace(x_vals.min() - 10, x_vals.max() + 10, 200)
y_fit = b_best + m_best * x_fit
plt.plot(x_fit, y_fit, color='darkred', linewidth=2, label='Best-fit line')

# Plain text (no backslashes for LaTeX)
line_eq = (
    f"y = ({b_best:.2f} ± {sigma_b:.2f}) + "
    f"({m_best:.2f} ± {sigma_m:.2f}) x"
)
extras = f"var(b)={var_b:.2f}, var(m)={var_m:.2f}"
annot_str = line_eq + "\n" + extras

plt.text(
    0.03, 0.90, annot_str,
    transform=plt.gca().transAxes, va='top', fontsize=10,
    bbox=dict(facecolor='whitesmoke', edgecolor='gray', alpha=0.7)
)

plt.title("Weighted Least-Squares Fit (ID=5..20)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.savefig("Ex1_without_latex.png", dpi=150)
plt.show()
