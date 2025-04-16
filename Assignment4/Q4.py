
"""
Created on Tue Apr 10 20:17:19 2025

@author: Salar
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import itertools


show_figures = True


with open('timeseries.pkl', 'rb') as f:
    times, ys, ivars = pickle.load(f)


y = ys[7]


sidereal_year = 365.256363004  # days
sidereal_day = sidereal_year / (sidereal_year + 1.)
omega_fixed = 2 * np.pi / sidereal_day  # base (fixed) frequency


def model_7(params, t):
    """
    7-parameter model:
      y(t) = mu + sum_{j=1}^{3} [a_j*cos(j*omega*t) + b_j*sin(j*omega*t)]
    params = [mu, a1, b1, a2, b2, a3, b3]
    """
    mu = params[0]
    a1, b1 = params[1], params[2]
    a2, b2 = params[3], params[4]
    a3, b3 = params[5], params[6]
    return (mu +
            a1*np.cos(omega_fixed*t) + b1*np.sin(omega_fixed*t) +
            a2*np.cos(2*omega_fixed*t) + b2*np.sin(2*omega_fixed*t) +
            a3*np.cos(3*omega_fixed*t) + b3*np.sin(3*omega_fixed*t))

def log_post_7(params, t, y, ivars):
    """
    Log-posterior for the 7-parameter model.
    Priors:
       mu ~ Uniform(0.5, 1.5)
       a_j, b_j ~ Uniform(-2, 2) for j=1,2,3.
    """
    mu, a1, b1, a2, b2, a3, b3 = params
    # Check the priors
    if not (0.5 < mu < 1.5 and -2 < a1 < 2 and -2 < b1 < 2 and
            -2 < a2 < 2 and -2 < b2 < 2 and -2 < a3 < 2 and -2 < b3 < 2):
        return -np.inf
    model_val = model_7(params, t)
    # Gaussian likelihood (ignoring constant factors)
    loglike = -0.5 * np.sum(ivars * (y - model_val)**2)
    return loglike


def run_mcmc(log_post, initial, nsteps, proposal_std, t, y, ivars):
    ndim = len(initial)
    chain = np.zeros((nsteps, ndim))
    current = np.array(initial)
    current_logpost = log_post(current, t, y, ivars)
    chain[0] = current
    for i in range(1, nsteps):
        proposal = current + np.random.normal(scale=proposal_std, size=ndim)
        prop_logpost = log_post(proposal, t, y, ivars)
        if np.log(np.random.rand()) < prop_logpost - current_logpost:
            current = proposal
            current_logpost = prop_logpost
        chain[i] = current
    return chain


nsteps = 20000
burnin = 5000
proposal_std = np.array([0.005]*7)  # same proposal sigma for all parameters


initial_params = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

chain = run_mcmc(log_post_7, initial_params, nsteps, proposal_std, times, y, ivars)
chain = chain[burnin:]  # discard burn-in


thin_factor = 10
chain_thin = chain[::thin_factor]


param_names = ['mu', 'a1', 'b1', 'a2', 'b2', 'a3', 'b3']


pairs = list(itertools.combinations(range(7), 2))  # 7-choose-2 = 21 pairs

fig, axes = plt.subplots(3, 7, figsize=(20, 10))
for idx, (i, j) in enumerate(pairs):
    ax = axes[idx // 7, idx % 7]
    ax.scatter(chain_thin[:, i], chain_thin[:, j], s=5, c='blue', alpha=0.5)
    ax.set_xlabel(param_names[i], fontsize=8)
    ax.set_ylabel(param_names[j], fontsize=8)
    ax.tick_params(labelsize=8)
plt.tight_layout()
plt.suptitle("Pairwise 2D Scatter Plots (Thinned Chain) for the 7-Parameter Model", y=1.02)
plt.savefig("figures_7param_pairwise.png", bbox_inches='tight')
if show_figures:
    plt.show()
plt.close()


phase = (times % sidereal_day) / sidereal_day
sorted_idx = np.argsort(phase)
phase_sorted = phase[sorted_idx]
y_sorted = y[sorted_idx]
yerr = 1.0/np.sqrt(ivars)[sorted_idx]


np.random.seed(123)
sample_indices = np.random.choice(chain.shape[0], size=12, replace=False)
models = []
for idx in sample_indices:
    params_sample = chain[idx]
    model_vals = model_7(params_sample, times)
    models.append(model_vals[sorted_idx])

plt.figure(figsize=(8,6))
plt.errorbar(phase_sorted, y_sorted, yerr=yerr, fmt='ko', markersize=3, capsize=2, label='Data')

for m in models:
    plt.plot(phase_sorted, m, alpha=0.6, lw=1)
plt.xlabel("Phase (folded at sidereal period)")
plt.ylabel("Brightness")
plt.title("Folded Data with 12 Posterior Sampled Models (7-Parameter Model)")
plt.legend()
plt.savefig("figures_7param_best_fit_models.png", bbox_inches='tight')
if show_figures:
    plt.show()
plt.close()


fig, axes = plt.subplots(7, 1, figsize=(10, 14), sharex=True)
for i in range(7):
    axes[i].plot(chain[:, i], color='blue', alpha=0.7)
    axes[i].set_ylabel(param_names[i])
axes[-1].set_xlabel("Iteration")
plt.suptitle("Trace Plots for the 7-Parameter Model")
plt.savefig("figures_7param_trace.png", bbox_inches='tight')
if show_figures:
    plt.show()
plt.close()


def autocorr(x, lags=50):
    N = len(x)
    x = x - np.mean(x)
    result = np.correlate(x, x, mode='full')
    acf = result[result.size//2:] / result[result.size//2]
    return acf[:lags]

fig, axes = plt.subplots(7, 1, figsize=(10, 14), sharex=True)
for i in range(7):
    acf = autocorr(chain[:, i], lags=100)
    axes[i].plot(acf, marker='o', linestyle='-')
    axes[i].set_ylabel(param_names[i])
    axes[i].set_ylim([-0.1, 1.1])
axes[-1].set_xlabel("Lag")
plt.suptitle("Autocorrelation Plots for the 7-Parameter Model")
plt.savefig("figures_7param_autocorrelation.png", bbox_inches='tight')
if show_figures:
    plt.show()
plt.close()

print(" complete.")
print("Figures saved:")
