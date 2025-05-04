"""
Author : Salar
"""

import pickle, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib  import Path
from sklearn.decomposition import PCA
from sklearn.neighbors      import NearestNeighbors
from sklearn.metrics        import mean_squared_error


OUTDIR  = Path("").resolve()                 
OUTDIR.mkdir(parents=True, exist_ok=True)

TRAIN_F = "ap17_xpcont_train.pickle"
VAL_F   = "ap17_xpcont_validation.pickle"

MAX_K   = 30
PCA_K   = 5
RSTATE  = 0

train = pickle.load(open(TRAIN_F, "rb"))
val   = pickle.load(open(VAL_F,   "rb"))

X_train_full = np.hstack([train["bp_coef"], train["rp_coef"]])   
X_val_full   = np.hstack([val  ["bp_coef"], val  ["rp_coef"]])   

y_train = train["labels"][:, 1]          
y_val   = val  ["labels"][:, 1]


pca = PCA(n_components=PCA_K, random_state=RSTATE)
X_train_pca = pca.fit_transform(X_train_full)
X_val_pca   = pca.transform     (X_val_full)


def knn_metrics(Xtr, Xva, ytr, yva, max_k):
    nn = NearestNeighbors(n_neighbors=max_k, metric="euclidean",
                          algorithm="brute")
    nn.fit(Xtr)
    _, idx_mat = nn.kneighbors(Xva)           

    rows = []
    for k in range(1, max_k + 1):
        neigh_targets = ytr[idx_mat[:, :k]]   

        pred_mean   = neigh_targets.mean(axis=1)
        pred_median = np.median(neigh_targets, axis=1)

        mse_mean    = mean_squared_error(yva, pred_mean)
        mse_median  = mean_squared_error(yva, pred_median)
        mad_mean    = np.median(np.abs(pred_mean   - yva))
        mad_median  = np.median(np.abs(pred_median - yva))

        rows.append((k, mse_mean, mse_median, mad_mean, mad_median))

    return pd.DataFrame(rows, columns=[
        "k", "MSE_mean", "MSE_median", "MAD_mean", "MAD_median"
    ])


results = {
    "orig": knn_metrics(X_train_full, X_val_full, y_train, y_val, MAX_K),
    "pca" : knn_metrics(X_train_pca,  X_val_pca,  y_train, y_val, MAX_K),
}


def best_k(df, col):
    row = df.loc[df[col].idxmin()]
    return int(row["k"]), row[col]


for tag, df in results.items():
    print(f"\n Validation metrics ({tag}) for k = 1 … {MAX_K} \n")
    print(df.to_string(index=False, formatters={
        "k"        : "{:2d}".format,
        "MSE_mean" : "{:.4f}".format,
        "MSE_median": "{:.4f}".format,
        "MAD_mean" : "{:.4f}".format,
        "MAD_median": "{:.4f}".format
    }))

print("\n Optimal-k :")
for metric in ["MSE_mean", "MSE_median", "MAD_mean", "MAD_median"]:
    for tag, df in results.items():
        k_opt, val_opt = best_k(df, metric)
        print(f"{metric:11s} — {tag:4s}: k = {k_opt:2d}   value = {val_opt:.4f}")


def plot_curves(metric_col, label, agg, fname):
    """metric_col e.g. 'MAD_median', label text for y-axis."""
    fig = plt.figure()
    styles = {"orig": ("o-", "tab:blue"),
              "pca" : ("s-", "tab:orange")}

    
    for tag in ["orig", "pca"]:
        df      = results[tag]
        style,c = styles[tag]
        plt.plot(df["k"], df[metric_col], style, label=f"{tag}")
        k_opt, y_opt = best_k(df, metric_col)
        plt.scatter(k_opt, y_opt, c=c, zorder=5)
        plt.text   (k_opt + 0.4, y_opt, f"k={k_opt}", va="bottom", color=c)

    plt.xlabel("k (nearest neighbours)")
    plt.ylabel(label)
    plt.title(f"{label} vs k ({agg} aggregation)")
    plt.grid(); plt.legend()
    plt.tight_layout()
    fig.savefig(OUTDIR / fname, dpi=300)


plot_curves("MAD_median", "Median Absolute Deviation", "median",
            "mad_vs_k_median.png")
plot_curves("MAD_mean",   "Median Absolute Deviation", "mean",
            "mad_vs_k_mean.png")
plot_curves("MSE_median", "Mean-Squared Error",        "median",
            "mse_vs_k_median.png")
plot_curves("MSE_mean",   "Mean-Squared Error",        "mean",
            "mse_vs_k_mean.png")


plt.show()        
