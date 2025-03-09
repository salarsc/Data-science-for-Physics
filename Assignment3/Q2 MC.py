
"""
Created on Sat Mar  8 11:32:54 2025

@author: Salar
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import os


with open('timeseries.pkl', 'rb') as f:
    times, ys, ivars = pickle.load(f)


sidereal_year = 365.256363004  # days
sidereal_day = sidereal_year / (sidereal_year + 1.)

omega0 = 2 * np.pi / sidereal_day  

omega_fixed = omega0

N_data = len(times)



def log_post_3(params, t, y, ivars):
    
    mu, a, b = params
    # Uniform prior check:
    if not (0.5 < mu < 1.5 and -2 < a < 2 and -2 < b < 2):
        return -np.inf
    # Model with fixed frequency:
    model = mu + a * np.cos(omega_fixed * t) + b * np.sin(omega_fixed * t)
    # Gaussian likelihood (ignoring constant factors)
    loglike = -0.5 * np.sum(ivars * (y - model)**2)
    return loglike

def log_post_4(params, t, y, ivars):
   
    mu, a, b, omega = params
    if not (0.5 < mu < 1.5 and -2 < a < 2 and -2 < b < 2 and 0.9 * omega0 < omega < 1.1 * omega0):
        return -np.inf
    model = mu + a * np.cos(omega * t) + b * np.sin(omega * t)
    loglike = -0.5 * np.sum(ivars * (y - model)**2)
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
        # Accept or reject:
        if np.log(np.random.rand()) < prop_logpost - current_logpost:
            current = proposal
            current_logpost = prop_logpost
        chain[i] = current
    return chain


def plot_2d_histograms(chain, param_names, pairs, filename, title):
    
    n_plots = len(pairs)
    
    ncols = n_plots
    nrows = 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4))
    if n_plots == 1:
        axes = [axes]
    for ax, (i, j) in zip(axes, pairs):
        h = ax.hist2d(chain[:, i], chain[:, j], bins=50, cmap='viridis')
        ax.set_xlabel(param_names[i])
        ax.set_ylabel(param_names[j])
        ax.set_title(f'{param_names[i]} vs {param_names[j]}')
        plt.colorbar(h[3], ax=ax)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(filename)
    plt.close(fig)


nsteps = 5000
burnin = 1000 
proposal_std_3 = np.array([0.005, 0.005, 0.005])

proposal_std_4 = np.array([0.005, 0.005, 0.005, 0.0005])


if not os.path.exists("figures_3param"):
    os.makedirs("figures_3param")
if not os.path.exists("figures_4param"):
    os.makedirs("figures_4param")


n_curves = ys.shape[0]

for i in range(n_curves):
    y = ys[i]
    
    print(f"\nLight Curve {i} Results:")
    
    
    mu_init = np.sum(y * ivars) / np.sum(ivars)
    initial_3 = [mu_init, 0.0, 0.0]
    
    chain_3 = run_mcmc(log_post_3, initial_3, nsteps, proposal_std_3, times, y, ivars)
   
    chain_3 = chain_3[burnin:]
    
    
    best_fit_3 = np.median(chain_3, axis=0)
    lower_3 = np.percentile(chain_3, 16, axis=0)
    upper_3 = np.percentile(chain_3, 84, axis=0)
    print("3-parameter model (fixed frequency):")
    print(f"  mu   = {best_fit_3[0]:.5f}  [{lower_3[0]:.5f}, {upper_3[0]:.5f}]")
    print(f"  a    = {best_fit_3[1]:.5f}  [{lower_3[1]:.5f}, {upper_3[1]:.5f}]")
    print(f"  b    = {best_fit_3[2]:.5f}  [{lower_3[2]:.5f}, {upper_3[2]:.5f}]")
    
   
    param_names_3 = ['mu', 'a', 'b']
    pairs_3 = [(0,1), (0,2), (1,2)]
    title_3 = f'Light Curve {i} (3-parameter model)'
    filename_3 = f'figures_3param/LC{i}_3param.png'
    plot_2d_histograms(chain_3, param_names_3, pairs_3, filename_3, title_3)
    
   
    initial_4 = [mu_init, 0.0, 0.0, omega0]
    
    chain_4 = run_mcmc(log_post_4, initial_4, nsteps, proposal_std_4, times, y, ivars)
    chain_4 = chain_4[burnin:]
    
    best_fit_4 = np.median(chain_4, axis=0)
    lower_4 = np.percentile(chain_4, 16, axis=0)
    upper_4 = np.percentile(chain_4, 84, axis=0)
    print("4-parameter model (frequency free):")
    print(f"  mu    = {best_fit_4[0]:.5f}  [{lower_4[0]:.5f}, {upper_4[0]:.5f}]")
    print(f"  a     = {best_fit_4[1]:.5f}  [{lower_4[1]:.5f}, {upper_4[1]:.5f}]")
    print(f"  b     = {best_fit_4[2]:.5f}  [{lower_4[2]:.5f}, {upper_4[2]:.5f}]")
    print(f"  omega = {best_fit_4[3]:.5f}  [{lower_4[3]:.5f}, {upper_4[3]:.5f}]")
    
    
    param_names_4 = ['mu', 'a', 'b', 'omega']
    
    pairs_4 = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    title_4 = f'Light Curve {i} (4-parameter model)'
    filename_4 = f'figures_4param/LC{i}_4param.png'
    
    
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    for ax, (p, q) in zip(axes.flatten(), pairs_4):
        h = ax.hist2d(chain_4[:, p], chain_4[:, q], bins=50, cmap='viridis')
        ax.set_xlabel(param_names_4[p])
        ax.set_ylabel(param_names_4[q])
        ax.set_title(f'{param_names_4[p]} vs {param_names_4[q]}')
        plt.colorbar(h[3], ax=ax)
    fig.suptitle(title_4)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(filename_4)
    plt.close(fig)

print("MCMC and plotting completed. Figures saved in 'figures_3param' and 'figures_4param'.")
