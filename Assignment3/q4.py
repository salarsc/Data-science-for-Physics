"""
Created on Sat Mar  8 21:38:45 2025

@author: Salar
"""

import numpy as np
import matplotlib.pyplot as plt
import emcee
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
x = data[:, 0]
y = data[:, 1]
sigma_y = data[:, 2]


def log_prob(theta, x, y, sigma):
    m, b, Pb, Yb, Vb = theta
    # Prior checks
    if not (0 < m < 5): return -np.inf
    if not (-50 < b < 150): return -np.inf
    if not (0 <= Pb <= 1): return -np.inf
    if not (250 < Yb < 600): return -np.inf
    if not (0 <= Vb < 1000): return -np.inf

    logL = 0.0
    for xi, yi, si in zip(x, y, sigma):
        # Inlier likelihood:
        L_in = (1 - Pb) / np.sqrt(2 * np.pi * si**2) * np.exp(- (yi - (m*xi + b))**2 / (2*si**2))
        # Outlier likelihood:
        L_out = Pb / np.sqrt(2 * np.pi * (si**2 + Vb)) * np.exp(- (yi - Yb)**2 / (2*(si**2 + Vb)))
        L = L_in + L_out
        if L <= 0:
            return -np.inf
        logL += np.log(L)
    return logL  # Uniform priors contribute constant


ndim = 5
nwalkers = 32

initial = np.array([2.0, 40.0, 0.1, np.median(y), 100.0])

p0 = initial + 1e-4 * np.random.randn(nwalkers, ndim)


sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(x, y, sigma_y))

nsteps = 20000
print("Running emcee sampler on original uncertainties...")
sampler.run_mcmc(p0, nsteps, progress=True)


flat_samples = sampler.get_chain(discard=5000, flat=True)
flat_logprob = sampler.get_log_prob(discard=5000, flat=True)


idx_map = np.argmax(flat_logprob)
MAP_sample = flat_samples[idx_map]
m_MAP, b_MAP, Pb_MAP, Yb_MAP, Vb_MAP = MAP_sample
print("MAP sample (original uncertainties):")
print("  m =", m_MAP)
print("  b =", b_MAP)
print("  Pb =", Pb_MAP)
print("  Yb =", Yb_MAP)
print("  Vb =", Vb_MAP)


if not os.path.exists("figures_emcee"):
    os.makedirs("figures_emcee")


mb_samples = flat_samples[:, :2]
plt.figure(figsize=(6,5))
plt.hist2d(mb_samples[:,0], mb_samples[:,1], bins=50, cmap="viridis")
plt.xlabel("m")
plt.ylabel("b")
plt.title("2D Histogram of (m,b) [emcee]")
plt.colorbar(label="Counts")
plt.tight_layout()
plt.savefig("figures_emcee/mb_hist_emcee.png")
plt.close()


plt.figure(figsize=(7,6))
plt.errorbar(x, y, yerr=sigma_y, fmt='ko', label="Data")
x_line = np.linspace(np.min(x)-5, np.max(x)+5, 100)

plt.plot(x_line, m_MAP * x_line + b_MAP, 'r-', lw=2, label="MAP line")

indices = np.random.choice(len(mb_samples), size=10, replace=False)
for idx in indices:
    m_s, b_s = mb_samples[idx]
    plt.plot(x_line, m_s * x_line + b_s, color="grey", alpha=0.4)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Mixture Model Fit (emcee): Data, MAP line, and 10 Posterior Samples")
plt.legend()
plt.tight_layout()
plt.savefig("figures_emcee/data_MAP_emcee.png")
plt.close()


Pb_samples = flat_samples[:, 2]
plt.figure(figsize=(6,5))
plt.hist(Pb_samples, bins=50, density=True, color="skyblue", edgecolor="k")
plt.xlabel("Pb")
plt.ylabel("Posterior density")
plt.title("Marginalized Posterior for Pb [emcee] (original σ)")
plt.tight_layout()
plt.savefig("figures_emcee/Pb_posterior_emcee.png")
plt.close()


sigma_y_reduced = sigma_y / 2.0
p0_reduced = initial + 1e-4 * np.random.randn(nwalkers, ndim)
sampler_reduced = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(x, y, sigma_y_reduced))
print("Running emcee sampler on uncertainties reduced by factor 2...")
sampler_reduced.run_mcmc(p0_reduced, nsteps, progress=True)
flat_samples_reduced = sampler_reduced.get_chain(discard=5000, flat=True)
Pb_samples_reduced = flat_samples_reduced[:, 2]
plt.figure(figsize=(6,5))
plt.hist(Pb_samples_reduced, bins=50, density=True, color="salmon", edgecolor="k")
plt.xlabel("Pb")
plt.ylabel("Posterior density")
plt.title("Marginalized Posterior for Pb [emcee] (σ reduced by 2)")
plt.tight_layout()
plt.savefig("figures_emcee/Pb_posterior_emcee_reduced.png")
plt.close()

print("emcee sampling for Exercises 6 and 7 completed (with both original and reduced uncertainties).")
