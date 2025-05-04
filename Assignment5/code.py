"""
Created on Fri May  2  2025
@author: Salar
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
from scipy.stats import ks_2samp
import pandas as pd
OUTDIR = Path("")
OUTDIR.mkdir(exist_ok=True)
PERCENTILE_CLIP = (1, 99)      
POINT_SIZE      = 4
with open('ap17_xpcont_train.pickle', 'rb') as f_tr:
    data_tr = pickle.load(f_tr)
with open('ap17_xpcont_validation.pickle', 'rb') as f_val:
    data_val = pickle.load(f_val)
X_train = np.hstack([data_tr['bp_coef'], data_tr['rp_coef']])
X_val   = np.hstack([data_val['bp_coef'], data_val['rp_coef']])
pca = PCA(n_components=5, random_state=0)
scores_tr = pca.fit_transform(X_train)
scores_val = pca.transform(X_val)
explained = pca.explained_variance_ratio_
for i, comp in enumerate(pca.components_, 1):
    fig, ax = plt.subplots()
    ax.plot(comp)
    ax.set_xlabel("Coefficient index")
    ax.set_ylabel("Component value")
    ax.set_title(f"Principal Component {i}")  
    fig.tight_layout()
    fname = OUTDIR / f"eigenspectrum_PC{i}.png"
    fig.savefig(fname, dpi=150)
    plt.show()
ratio_labels = [r"$a_1/a_2$", r"$a_1/a_3$", r"$a_1/a_4$", r"$a_1/a_5$"]
def make_ratios(scores):
    
    return np.column_stack([scores[:, 0] / scores[:, j] for j in range(1, 5)])
def scatter_set(ratios, labels, colour, setname):
    k = ratios.shape[1]
    pairs = [(i, j) for i in range(k) for j in range(i+1, k)]
    for (i, j) in pairs:
        
        fig, ax = plt.subplots()
        sc = ax.scatter(ratios[:, i], ratios[:, j],
                        c=colour, s=POINT_SIZE, cmap="viridis")
        ax.set_xlabel(labels[i]); ax.set_ylabel(labels[j])
        ax.set_title(f"{setname}: {labels[i]} vs {labels[j]}")
        fig.colorbar(sc, label=r"$T_{\rm eff}\;[K]$")
        fig.tight_layout()
        fname = OUTDIR / f"{setname}_{i}{j}_full.png"
        fig.savefig(fname, dpi=150)
        plt.show()
        
        x, y = ratios[:, i], ratios[:, j]
        good = np.isfinite(x) & np.isfinite(y)
        if good.sum() == 0:
            continue
        p_lo, p_hi = np.percentile(x[good], PERCENTILE_CLIP), np.percentile(y[good], PERCENTILE_CLIP)
        fig, ax = plt.subplots()
        sc = ax.scatter(x, y, c=colour, s=POINT_SIZE, cmap="viridis")
        ax.set_xlim(*p_lo); ax.set_ylim(*p_hi)
        ax.set_xlabel(labels[i]); ax.set_ylabel(labels[j])
        ax.set_title(f"{setname} (zoom): {labels[i]} vs {labels[j]}")
        fig.colorbar(sc, label=r"$T_{\rm eff}\;[K]$")
        fig.tight_layout()
        fname = OUTDIR / f"{setname}_{i}{j}_zoom.png"
        fig.savefig(fname, dpi=150)
        plt.show()
ratios_tr  = make_ratios(scores_tr)
ratios_val = make_ratios(scores_val)
scatter_set(ratios_tr,  ratio_labels, data_tr['labels'][:, 0], "Training")
scatter_set(ratios_val, ratio_labels, data_val['labels'][:, 0], "Validation")
def make_ratios(scores):
    return np.column_stack([scores[:,0] / scores[:,j] for j in range(1,5)])
rat_tr  = make_ratios(scores_tr)
rat_val = make_ratios(scores_val)
rows = []
ratio_labels = ["a1/a2","a1/a3","a1/a4","a1/a5"]
for k,lbl in enumerate(ratio_labels):
    μ_tr, σ_tr  = np.nanmean(rat_tr[:,k]),  np.nanstd(rat_tr[:,k])
    μ_val,σ_val = np.nanmean(rat_val[:,k]), np.nanstd(rat_val[:,k])
    ks_stat, ks_p = ks_2samp(rat_tr[:,k], rat_val[:,k])
    rows.append([lbl, μ_tr, σ_tr, μ_val, σ_val, ks_stat, ks_p])
df = pd.DataFrame(rows, columns=
        ["ratio","μ_train","σ_train","μ_val","σ_val","KS-stat","KS p"])
print(df.to_string(index=False))