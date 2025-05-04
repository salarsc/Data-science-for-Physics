
"""
Created on Sun May  4 01:38:28 2025

@author: Salar
"""

import pickle, os, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import itertools, shutil


train_path = 'ap17_xpcont_train.pickle'
val_path   = 'ap17_xpcont_validation.pickle'
out_dir    = 'kmeans_hr_plots_s1'
os.makedirs(out_dir, exist_ok=True)


with open(train_path,'rb') as f:
    train = pickle.load(f)
with open(val_path,'rb') as f:
    val = pickle.load(f)

X_train = np.hstack([train['bp_coef'], train['rp_coef']])
y_train_Teff, y_train_logg = train['labels'][:,0], train['labels'][:,1]

X_val   = np.hstack([val['bp_coef'], val['rp_coef']])
y_val_Teff,   y_val_logg   = val['labels'][:,0],   val['labels'][:,1]


def scatter_hr(T, logg, labels, title, filename, cmap='tab10', dot_size=1):
    uniq = np.unique(labels)
    cmap_obj = plt.get_cmap(cmap)
    colors   = cmap_obj(labels % cmap_obj.N)
    plt.figure(figsize=(6.5,4.5))
    plt.scatter(T, logg, c=colors, s=dot_size, alpha=0.8)
    plt.xlabel('Teff [K]  (cool → hot)')
    plt.ylabel('log g [dex]')
    plt.title(title)
    handles = [Patch(facecolor=cmap_obj(i % cmap_obj.N), label=f'cluster {i}') for i in uniq]
    plt.legend(handles=handles, title='k-means id', loc='best', fontsize=8, frameon=True)
    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=300)
    plt.show()
    


cluster_labels = {}
for k in [2,3,4]:
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = km.fit_predict(X_train)
    cluster_labels[k]=labels
    scatter_hr(y_train_Teff, y_train_logg, labels,
               title=f'K-means (k={k}) – Training set (s=1)',
               filename=f'k{k}_train_s1.png',
               dot_size=1)


seeds=[0,1,2,3,4]
labs=[]
for seed in seeds:
    km=KMeans(n_clusters=3,n_init=1,random_state=seed).fit(X_train)
    labs.append(km.labels_)
ari=np.zeros((5,5))
for i,j in itertools.product(range(5),range(5)):
    ari[i,j]=adjusted_rand_score(labs[i],labs[j])
print("ARI matrix k=3 (s=1):")
print(np.round(ari,3))


km3 = KMeans(n_clusters=3,n_init=10,random_state=0).fit(X_train)
val_labels=km3.predict(X_val)
scatter_hr(y_val_Teff,y_val_logg,val_labels,
           title='K-means (k=3) – Validation set (s=1)',
           filename='k3_validation_s1.png',
           dot_size=1)
