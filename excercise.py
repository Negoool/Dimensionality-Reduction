''' exercises of chapter 8'''
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.decomposition import  PCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import os
os.system('clear')

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
print "starts"
np.random.seed(42)
m = 5000
idx = np.random.permutation(60000)[:m]
X = X_mnist[idx]
y = y_mnist[idx]

# tsne = TSNE(n_components=2,random_state = 42)
# X_reduced = tsne.fit_transform(X)
# plt.figure()
# plt.title("visualizig subset of mnist using tsne dimension reduction")
# plt.scatter(X_reduced[:,0], X_reduced[:,1], c = y, cmap = plt.get_cmap('jet'))
# plt.colorbar()

# plt.figure()
# cmap = plt.get_cmap('jet')
# for i in [2,3,5]:
#     plt.scatter(X_reduced[y==i,0], X_reduced[y==i,1], c=cmap(i / 9.))
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

def plot_digit(X_reduced_0, y, X = None, min_dis = .6):
    plt.figure()
    # normalize data
    X_reduced= MinMaxScaler().fit_transform(X_reduced_0)
    # points that we are going to plot their digit
    # far point for start
    plot_points = np.array([[100,100]])
    cmap = plt.get_cmap('jet')
    # first like before plot scatter plot
    plt.scatter(X_reduced[:,0], X_reduced[:,1], c = y, cmap = plt.get_cmap('jet'))
    plt.colorbar()
    # on the same axis
    ax = plt.gcf().gca()
    for index in range(len(X_reduced)):
        # if a point is far away all points that are going to be plotted
        # add it to the list and then add its digit
        if (euclidean_distances(X_reduced[index:index+1], plot_points)).min() > min_dis:
            plot_points = np.r_[plot_points, X_reduced[index: index+1]]

            if X is None:
                plt.text(X_reduced[index:index+1,0], X_reduced[index:index+1,1],
                y[index], color = cmap(y[index]/ 9.) ,
                fontdict={"weight": "bold", "size": 16})
            else:
                image = X[index].reshape(28,28)
                imagebox = AnnotationBbox(\
                OffsetImage(image,zoom=.6, cmap = 'binary'), X_reduced[index],
                pad = .1)
                ax.add_artist(imagebox )

tsne = TSNE(n_components=2,random_state = 42)
pca = PCA(n_components=2, random_state = 42)
kernel_pca = KernelPCA(n_components=2, kernel='cosine',random_state =42)
lda = LinearDiscriminantAnalysis(n_components=2)
lle = LocallyLinearEmbedding(n_components=2, random_state = 42)
mds = MDS(n_components=2, random_state = 42)
isomap = Isomap(n_components=2)


for technique in (pca, kernel_pca,  lle ,lda, isomap, tsne):
    s1 = time.time()
    X_reduced = technique.fit_transform(X, y)
    s2 = time.time()
    print "time takes for " + str(technique.__class__.__name__) +"\t"+ str(s2 - s1)
    plot_digit(X_reduced, y)
    plt.title(technique.__class__.__name__)


pca = PCA(n_components=.95, random_state = 42)
pca_tsne = Pipeline([('PCA',pca),('t_SNE', tsne)])
s1 = time.time()
X_reduced = pca_tsne.fit_transform(X)
s2 = time.time()
print "pca + tsne take time", (s2 - s1)
plot_digit(X_reduced, y)

plt.show()
