''' Locality Linear Embedding'''
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import Isomap
from sklearn.manifold import MDS
from sklearn.manifold import TSNE

# create toy data set
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=41)
# visualize it
fig = plt.figure()
# ax = Axes3D(fig)
# ax = fig.add_subplot(111, projection='3d')
# ax.set_title("Original data")
# ax.scatter(X[:,0], X[:,1], X[:,2], c = t, cmap=plt.cm.Spectral)


## reduce dim with lle
LLE = LocallyLinearEmbedding(n_neighbors=10, n_components=2, random_state =42)
X_reduced = LLE.fit_transform(X)
print "reconstuction error :", LLE.reconstruction_error_
# visualizing manifold after applying lle
plt.figure()
plt.title("Unrolled swiss roll using LLE")
plt.xlabel("$z_1$", fontsize = 18)
plt.ylabel("$z_2$", fontsize = 18)
plt.scatter(X_reduced[:,0], X_reduced[:,1], c = t, cmap=plt.cm.Spectral)


## reduce dim with isomap
iso = Isomap(n_neighbors=10, n_components=2)
X_reduced = iso.fit_transform(X)
# visualizing manifold after applying isomap
plt.figure()
plt.title("Unrolled swiss roll using isomap")
plt.xlabel("$z_1$", fontsize = 18)
plt.ylabel("$z_2$", fontsize = 18)
plt.scatter(X_reduced[:,0], X_reduced[:,1], c = t, cmap=plt.cm.Spectral)


## reduce dim with Mulidimentional Scaling(MDS)
mds = MDS(n_components=2, random_state = 42)
X_reduced = mds.fit_transform(X)
# visualizing manifold after applying MDS
plt.figure()
plt.title("Unrolled swiss roll using MDS")
plt.xlabel("$z_1$", fontsize = 18)
plt.ylabel("$z_2$", fontsize = 18)
plt.scatter(X_reduced[:,0], X_reduced[:,1], c = t, cmap=plt.cm.Spectral)

## reduce dim with t-SNE
tsne = TSNE(n_components=2, random_state = 42)
X_reduced = tsne.fit_transform(X)
# visualizing manifold after applying MDS
plt.figure()
plt.title("Unrolled swiss roll using t-SNE")
plt.xlabel("$z_1$", fontsize = 18)
plt.ylabel("$z_2$", fontsize = 18)
plt.scatter(X_reduced[:,0], X_reduced[:,1], c = t, cmap=plt.cm.Spectral)

plt.show()
