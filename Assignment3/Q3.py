"""
Created on Sat Mar  8 12:20:39 2025

@author: Salar
"""
print("started...")
import numpy as np
import matplotlib.pyplot as plt
import os


data = np.array([
    [203, 495, 21],
    [58,  173, 15],
    [210, 479, 27],
    [202, 504, 14],
    [198, 510, 30],
    [158, 416, 16],
    [165, 393, 14],
    [201, 442, 25],
    [157, 317, 52],
    [131, 311, 16],
    [166, 400, 34],
    [160, 337, 31],
    [186, 423, 42],
    [125, 334, 26],
    [218, 533, 16],
    [146, 344, 22]
])
x = data[:,0]
y = data[:,1]
sigma_y = data[:,2]


def log_post_mix(theta, x, y, sigma):
    m, b, Pb, Yb, Vb = theta
    # Check priors:
    if not (0 < m < 5): return -np.inf
    if not (-50 < b < 150): return -np.inf
    if not (0 <= Pb <= 1): return -np.inf
    if not (250 < Yb < 600): return -np.inf
    if not (0 <= Vb < 1000): return -np.inf
    
    logL = 0.0
    for xi, yi, si in zip(x, y, sigma):
        # Inlier: mean = m*xi + b, variance = si^2
        inlier = (1 - Pb) / np.sqrt(2 * np.pi * si**2) * np.exp(- (yi - (m*xi+b))**2 / (2*si**2))
        # Outlier: mean = Yb, variance = si^2 + Vb
        outlier = Pb / np.sqrt(2 * np.pi * (si**2 + Vb)) * np.exp(- (yi - Yb)**2 / (2*(si**2+Vb)))
        like = inlier + outlier
        # Prevent log(0)
        if like <= 0:
            return -np.inf
        logL += np.log(like)
    return logL  # uniform priors contribute constant (zero)


def run_mcmc(log_post, initial, nsteps, proposal_std, x, y, sigma):
    ndim = len(initial)
    chain = np.zeros((nsteps, ndim))
    current = np.array(initial)
    current_logpost = log_post(current, x, y, sigma)
    chain[0] = current
    for i in range(1, nsteps):
        proposal = current + np.random.normal(scale=proposal_std, size=ndim)
        prop_logpost = log_post(proposal, x, y, sigma)
        if np.log(np.random.rand()) < prop_logpost - current_logpost:
            current = proposal
            current_logpost = prop_logpost
        chain[i] = current
    return chain


nsteps = 20000
burnin = 5000
proposal_std = np.array([0.01, 0.5, 0.01, 0.5, 1.0])  # tuned proposal widths


m_init = 2.0
b_init = 40.0
initial = [m_init, b_init, 0.1, np.median(y), 100.0]  # Pb small, Yb ~ median(y), Vb moderate

chain = run_mcmc(log_post_mix, initial, nsteps, proposal_std, x, y, sigma_y)
chain = chain[burnin:]  # discard burn-in


logposts = np.array([log_post_mix(sample, x, y, sigma_y) for sample in chain])
idx_MAP = np.argmax(logposts)
MAP_sample = chain[idx_MAP]
m_MAP, b_MAP, Pb_MAP, Yb_MAP, Vb_MAP = MAP_sample
print("Mixture model MAP sample:")
print("  m =", m_MAP)
print("  b =", b_MAP)
print("  Pb =", Pb_MAP)
print("  Yb =", Yb_MAP)
print("  Vb =", Vb_MAP)


mb_chain = chain[:, 0:2]


if not os.path.exists("figures_mix"):
    os.makedirs("figures_mix")


plt.figure(figsize=(6,5))
plt.hist2d(mb_chain[:,0], mb_chain[:,1], bins=50, cmap="viridis")
plt.xlabel("m")
plt.ylabel("b")
plt.title("2D Histogram of (m, b) from Mixture Model MCMC")
plt.colorbar(label="Counts")
plt.tight_layout()
plt.savefig("figures_mix/mb_hist.png")
plt.close()


plt.figure(figsize=(7,6))
plt.errorbar(x, y, yerr=sigma_y, fmt='ko', label="Data")

x_line = np.linspace(np.min(x)-5, np.max(x)+5, 100)

y_MAP_line = m_MAP * x_line + b_MAP
plt.plot(x_line, y_MAP_line, 'r-', lw=2, label="MAP line")

indices = np.random.choice(len(mb_chain), size=10, replace=False)
for idx in indices:
    m_sample, b_sample = mb_chain[idx]
    plt.plot(x_line, m_sample*x_line + b_sample, color="grey", alpha=0.4)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Mixture Model Fit: Data, MAP Line, and 10 Posterior Samples")
plt.legend()
plt.tight_layout()
plt.savefig("figures_mix/data_MAP_lines.png")
plt.close()



Pb_chain = chain[:, 2]
plt.figure(figsize=(6,5))
plt.hist(Pb_chain, bins=50, density=True, color="skyblue", edgecolor="k")
plt.xlabel("Pb")
plt.ylabel("Posterior density")
plt.title("Marginalized Posterior for Pb (Original uncertainties)")
plt.tight_layout()
plt.savefig("figures_mix/Pb_posterior.png")
plt.close()



sigma_y_reduced = sigma_y / 2.0  

chain_reduced = run_mcmc(log_post_mix, initial, nsteps, proposal_std, x, y, sigma_y_reduced)
chain_reduced = chain_reduced[burnin:]
Pb_chain_reduced = chain_reduced[:,2]
plt.figure(figsize=(6,5))
plt.hist(Pb_chain_reduced, bins=50, density=True, color="salmon", edgecolor="k")
plt.xlabel("Pb")
plt.ylabel("Posterior density")
plt.title("Marginalized Posterior for Pb (σ_y reduced by factor 2)")
plt.tight_layout()
plt.savefig("figures_mix/Pb_posterior_reduced.png")
plt.close()

print("completed.")
