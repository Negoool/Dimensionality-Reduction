''' exercises of chapter 8'''
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import  PCA
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
X_mnist = mnist['data']
y_mnist = mnist['target']
X_train, X_test = X_mnist[:60000], X_mnist[-10000:]
y_train,  y_test = y_mnist[:60000], y_mnist[-10000:]

## excercise 9
# rnd_clf = RandomForestClassifier(n_estimators=10, max_features = 'auto',
# random_state =42)
# s1 = time.time()
# rnd_clf.fit(X_train, y_train)
# s2 = time.time()
# print "time of training rdforest on original data",(s2 - s1)
# print "accuracy of model trained on original data",rnd_clf.score(X_test, y_test)
#
# pca = PCA(n_components = .95, random_state =42)
# s3 = time.time()
# X_reduced = pca.fit_transform(X_train)
# s4 = time.time()
# print "time for perform PCA", s4 - s3
# rnd_clf = RandomForestClassifier(n_estimators=10, max_features = 'auto',
# random_state =42)
# s5 = time.time()
# rnd_clf.fit(X_reduced, y_train)
# s6 = time.time()
# print "time of training rdforest on reduced data",(s6 - s5)
# X_reduced_test = pca.transform(X_test)
# print "accuracy of model trained on reduced data",\
# rnd_clf.score(X_reduced_test, y_test)
#
# print " softmax"
# log_clf = LogisticRegression(multi_class="multinomial", solver="lbfgs",
#  random_state=42)
# s1 = time.time()
# log_clf.fit(X_train, y_train)
# s2 = time.time()
# print "time of training rdforest on original data",(s2 - s1)
# print "accuracy of model trained on original data",log_clf.score(X_test, y_test)
#
# log_clf = LogisticRegression(multi_class="multinomial", solver="lbfgs",
#  random_state=42)
# s5 = time.time()
# log_clf.fit(X_reduced, y_train)
# s6 = time.time()
# print "time of training rdforest on reduced data",(s6 - s5)
# print "accuracy of model trained on reduced data",\
# log_clf.score(X_reduced_test, y_test)

## exercise 10
np.random.seed(42)
m = 10000
idx = np.random.permutation(60000)[:m]
X = X_mnist[idx]
y = y_mnist[idx]

tsne = TSNE(n_components=2,random_state = 42)
X_reduced = tsne.fit_transform(X)
plt.figure()
plt.title("visualizig mnist data set using tsne dimension reduction")
plt.scatter(X_reduced[:,0], X_reduced[:,1], c = y, cmap = "paired")
plt.show()
