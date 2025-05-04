
"""
Author : Salar
"""

import pickle, numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics  import mean_squared_error
import matplotlib.pyplot as plt


train = pickle.load(open('ap17_xpcont_train.pickle', 'rb'))
val   = pickle.load(open('ap17_xpcont_validation.pickle', 'rb'))

X_train = np.hstack([train['bp_coef'], train['rp_coef']])        
X_val   = np.hstack([val  ['bp_coef'], val  ['rp_coef']])        
y_train = train['labels'][:, 1]                                  
y_val   = val  ['labels'][:, 1]


max_k = 30
nn = NearestNeighbors(n_neighbors=max_k, metric='euclidean', algorithm='brute')
nn.fit(X_train)
_, inds = nn.kneighbors(X_val)                                   


rows = []
for k in range(1, max_k + 1):
    idx         = inds[:, :k]
    neigh_vals  = y_train[idx]

    pred_mean   = neigh_vals.mean(axis=1)
    pred_median = np.median(neigh_vals, axis=1)

    mse_mean    = mean_squared_error(y_val, pred_mean)
    mse_median  = mean_squared_error(y_val, pred_median)
    mad_mean    = np.median(np.abs(pred_mean   - y_val))
    mad_median  = np.median(np.abs(pred_median - y_val))

    rows.append((k, mse_mean, mse_median, mad_mean, mad_median))

results = pd.DataFrame(
    rows,
    columns=['k', 'MSE_mean', 'MSE_median', 'MAD_mean', 'MAD_median']
)

print('\n===== Validation metrics for k = 1 … 30 =====\n')
print(results.to_string(index=False, formatters={
      'k'         : '{:2d}'.format,
      'MSE_mean'  : '{:.4f}'.format,
      'MSE_median': '{:.4f}'.format,
      'MAD_mean'  : '{:.4f}'.format,
      'MAD_median': '{:.4f}'.format}))


best_mse_mean   = results.loc[results['MSE_mean'  ].idxmin(), ['k', 'MSE_mean'  ]]
best_mse_median = results.loc[results['MSE_median'].idxmin(), ['k', 'MSE_median']]
best_mad_mean   = results.loc[results['MAD_mean'  ].idxmin(), ['k', 'MAD_mean'  ]]
best_mad_median = results.loc[results['MAD_median'].idxmin(), ['k', 'MAD_median']]

print('\n===== Best-k summary =====')
print(f"MSE (mean):   k = {int(best_mse_mean['k']):2d}   MSE = {best_mse_mean['MSE_mean']:.4f}")
print(f"MSE (median): k = {int(best_mse_median['k']):2d}   MSE = {best_mse_median['MSE_median']:.4f}")
print(f"MAD (mean):   k = {int(best_mad_mean['k']):2d}   MAD = {best_mad_mean['MAD_mean']:.4f}")
print(f"MAD (median): k = {int(best_mad_median['k']):2d}   MAD = {best_mad_median['MAD_median']:.4f}")


fig1 = plt.figure()
plt.plot(results['k'], results['MSE_mean'  ], 'o-', label='Mean aggregation')
plt.plot(results['k'], results['MSE_median'], 's-', label='Median aggregation')

plt.scatter(best_mse_mean['k'], best_mse_mean['MSE_mean'],   c='k', zorder=5)
plt.text   (best_mse_mean['k']+0.5, best_mse_mean['MSE_mean'],
            f'k={int(best_mse_mean["k"])}', va='bottom')
plt.scatter(best_mse_median['k'], best_mse_median['MSE_median'], c='k', zorder=5)
plt.text   (best_mse_median['k']+0.5, best_mse_median['MSE_median'],
            f'k={int(best_mse_median["k"])}', va='bottom')

plt.xlabel('k (nearest neighbours)');  plt.ylabel('Mean-Squared Error')
plt.title('Validation MSE vs k');      plt.grid();  plt.legend()
plt.tight_layout();  fig1.savefig('mse_vs_k.png', dpi=300)

fig2 = plt.figure()
plt.plot(results['k'], results['MAD_mean'  ], 'o-', label='Mean aggregation')
plt.plot(results['k'], results['MAD_median'], 's-', label='Median aggregation')

plt.scatter(best_mad_mean['k'], best_mad_mean['MAD_mean'],   c='k', zorder=5)
plt.text   (best_mad_mean['k']+0.5, best_mad_mean['MAD_mean'],
            f'k={int(best_mad_mean["k"])}', va='bottom')
plt.scatter(best_mad_median['k'], best_mad_median['MAD_median'], c='k', zorder=5)
plt.text   (best_mad_median['k']+0.5, best_mad_median['MAD_median'],
            f'k={int(best_mad_median["k"])}', va='bottom')

plt.xlabel('k (nearest neighbours)');  plt.ylabel('Median Absolute Deviation')
plt.title('Validation MAD vs k');      plt.grid();  plt.legend()
plt.tight_layout();  fig2.savefig('mad_vs_k.png', dpi=300)

plt.show()
