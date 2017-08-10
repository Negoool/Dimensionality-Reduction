import numpy as np
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# create toy data set
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
# visualize it
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:,0], X[:,1], X[:,2], c = t, cmap = "hot")

# define an object of a rbf kernel pca
pca_rbf = KernelPCA(n_components=2, kernel='rbf', gamma=0.0433,
fit_inverse_transform=True, random_state=42)
# define an object of a linear kernel pca (simple pca)
pca_lin = KernelPCA(n_components=2, kernel='linear',
fit_inverse_transform=True, random_state=42)
# # define an object of a sigmoid kernel pca
pca_sig = KernelPCA(n_components=2, kernel='sigmoid', gamma=0.001, coef0=1,
fit_inverse_transform=True, random_state=42)

plt.figure()
for pca, title, sub_num in ((pca_lin,"Linear Kernel",131),\
(pca_rbf,"RBF Kernel",132),(pca_sig,"SigmoidKernel",133)):
    X_reduced = pca.fit_transform(X)
    plt.subplot(sub_num)
    plt.title(title)
    # visualize the reduced dimension(2D)
    plt.scatter(X_reduced[:,0], X_reduced[:,1], c=t, cmap="hot")
    plt.xlabel("$z_1$", fontsize = 18)
    plt.ylabel("$z_2$", fontsize = 18)

### finding best kernel and parameters
## approach1, preimage
rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=.04,
 fit_inverse_transform=True)
X_reduced_rbf = rbf_pca.fit_transform(X)
X_preimage = rbf_pca.inverse_transform(X_reduced_rbf)
print mean_squared_error(X, X_preimage)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X_preimage[:,0], X_preimage[:,1], X_preimage[:,2], c = t, cmap = "jet")

## approach2
y = t > 6.9
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:,0], X[:,1], X[:,2], c = y, cmap = "ocean")

pca = KernelPCA(n_components = 2, random_state=42)
log_clf = LogisticRegression(random_state = 42)
my_pip = Pipeline([('KernelPCA', pca), ('log_clf', log_clf)])
# param_grid = [\
# {"KernelPCA__kernel":["rbf"], "KernelPCA__gamma": np.linspace(.01,.09,50)},
# {"KernelPCA__kernel":["sigmoid"], "KernelPCA__gamma": np.linspace(.01,.09,10),\
# "KernelPCA__coef0": np.arange(0,5,1)}\
# ]
# grid =GridSearchCV(my_pip, param_grid= param_grid, cv=3, scoring = 'accuracy',
# verbose=0)
# grid.fit(X,y)
# print grid.best_params_
# my_estimator = grid.best_estimator_
my_pip.set_params(KernelPCA__kernel = 'rbf', KernelPCA__gamma = .0443)
my_pip.fit(X,y)
X_reduced = my_pip.named_steps['KernelPCA'].fit_transform(X)

plt.figure()
z1, z2 = np.meshgrid(
        np.linspace(-.8, .8, 200).reshape(-1, 1),
        np.linspace(-.8, .8, 200).reshape(-1, 1),
        )
Z_new = np.c_[z1.ravel(), z2.ravel()]
pred = my_pip.named_steps['log_clf']. predict(Z_new)

zz = pred.reshape(z1.shape)
plt.contourf(z1, z2, zz, cmap = "gnuplot", alpha = .1)
plt.scatter(X_reduced[:,0], X_reduced[:,1], c = y, cmap = 'ocean')

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:,0], X[:,1], X[:,2], c = my_pip.predict(X), cmap = "ocean")

plt.show()
