''' dimensionality reduction'''
# load data
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
X_mnist = mnist['data']
y_mnist = mnist['target']
X_train, X_test = X_mnist[:60000], X_mnist[-10000:]
print " X train shape", X_train.shape
y_train,  y_test = y_mnist[:60000], y_mnist[-10000:]
np.random.seed(42)
shuffled_indices = np.random.permutation(len(X_train))
X_train = X_train[shuffled_indices]
y_train = y_train[shuffled_indices]

# def plot_digit(data):
#     image = data.reshape((28,28))
#     plt.imshow(image, cmap = 'binary')
#     plt.colorbar()
#     plt.axis("off")
# plot_digit(X_train[20000])
# plt.title('one oroginal data')
#
# from sklearn.decomposition import PCA
# #select best k (reduction Dimension)
# #plot comulative explained_variance_ratio VS # of dim and use elbow to select k
# pca = PCA(random_state = 42)
# pca.fit(X_train)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# plt.figure()
# plt.plot(cumsum)
# plt.xlabel('Dimension')
# plt.ylabel('Expleined Variance')
#
# # another way to select k, just pass the desirable preserved variance
# pca = PCA(random_state = 42, n_components = .95)
# X_reduced = pca.fit_transform(X_train)
# print "number of components for .95 variance", pca.n_components_
#
#
# X_recovered = pca.inverse_transform(X_reduced)
# plt.figure()
# plot_digit(X_recovered[20000])
# plt.title('Compressed with PCA')
#
# ## comparision of time(PCA, INCREMENTAL PCA, RANDOMIZAD PCA)
# st1 = time.time()
# pca = PCA(random_state = 42, n_components = 154)
# pca.fit(X_train)
# X_reduced = pca.transform(X_train)
# sp1 = time.time()
# print "time for PCA", sp1 -st1
#
# from sklearn.decomposition import IncrementalPCA
# st2 = time.time()
# n_batches = 100
# inc_pca = IncrementalPCA(n_components = 154)
# for X_batch in np.array_split(X_train, n_batches):
#     inc_pca.partial_fit(X_batch)
# X_reduced2 = inc_pca.transform(X_batch)
# sp2 = time.time()
# print " incremental PCA time", sp2 - st2
#
# st3 = time.time()
# rnd_pca = PCA(n_components = 154, svd_solver="randomized", random_state=42)
# rnd_pca.fit(X_train)
# X_reduced = rnd_pca.transform(X_train)
# sp3 = time.time()
# print "time for randomized PCA", sp3 -st3

print "*"*10+"LDA"+"*"*10
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
# lda.fit(X_train, y_train)
# X_reduced_lda = lda.transform(X_train)
# print X_reduced_lda.shape
# y_pred = lda.predict(X_test)
# print accuracy_score(y_pred, y_test)
#
# from sklearn.svm import SVC
# svm_clf = SVC()
# svm_clf.fit(X_reduced_lda, y_train)
# X_test_reduced = lda.transform(X_test)
# y_pred = svm_clf.predict(X_test_reduced)
# print accuracy_score(y_pred, y_test)
#
knn_clf = KNeighborsClassifier()
from sklearn.pipeline import Pipeline
lda_knn =  Pipeline([('LDA', lda), ('KNN', knn_clf)])

from sklearn.model_selection import GridSearchCV
param_grid = [{"KNN__n_neighbors":np.arange(5,30,5)}]
grid_search = GridSearchCV(lda_knn, param_grid= param_grid, scoring='accuracy',
verbose = 2,  cv = 3)
grid_search.fit(X_train,y_train)
print grid_search.best_params_
