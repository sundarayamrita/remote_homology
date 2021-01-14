import numpy as np
import argparse
from pathlib import Path

pos_x = np.load('pos.npy')
neg_x = np.load('neg.npy')
#print("original x pos 35", pos_x[35])
#print("original x neg 132", neg_x[132])

pos_y = np.ones((pos_x.shape[0]))
neg_y = np.zeros((neg_x.shape[0]))
#print("original y pos 35", pos_y[35])
#print("original y neg 132", neg_y[132])

from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
rs = RandomState(MT19937(SeedSequence(123456789)))

X = np.vstack((pos_x, neg_x))
print("x shape", X.shape)
np.random.shuffle(X)
y = np.concatenate((pos_y, neg_y), axis = 0)
print("y shape", y.shape)
np.random.shuffle(y)

#print("shuffled x pos 35", pos_x[35])
#print("shuffled x neg 132", neg_x[132])
#print("shuffled y pos 35", pos_y[35])
#print("shuffled y neg 132", neg_y[132])

#from sklearn.svm import SVC
#from sklearn.datasets import load_iris
#from sklearn.model_selection import StratifiedShuffleSplit
#from sklearn.model_selection import GridSearchCV

#C_raange = np.logspace(-2, 10, 13)
#gamma_range = np.logspace(-9, 3, 13)
#param_grid = dict(gamma=gamma_range, C=C_range)
#cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
#grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
#grid.fit(X, y)

#print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))

#from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
l, m, n = X_train.shape
X_train = np.reshape(X_train, (l, m * n))
a, b, c = X_test.shape
X_test = np.reshape(X_test, (a, b * c))

clf = svm.SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
recall = tp / (tp + fn)
print("tp ", tp)
print("fp ", fp)
print("tn ", tn)
print("fn ", fn)
precision = tp / (tp + fp)
print("accuracy ", accuracy)
print("recall ", recall)
print("precision ", precision)

#rs = ShuffleSplit(n_splits=5, test_size=.20, random_state=0)

#for train_index, test_index in rs.split(X):
    
#    X_train, X_test = X[train_index], X[test_index]
#    l, m, n = X_train.shape
#    X_train = np.reshape(X_train, (l, m * n))
#    a, b, c = X_test.shape
#    X_test = np.reshape(X_test, (a, b * c))
#    y_train, y_test = y[train_index], y[test_index]
#    print("x_train shape", X_train.shape)
#    print("y_train shape", y_train.shape)
#    clf.fit(X_train, y_train)
#    y_pred = clf.predict(X_test)
#    print("y_pred shape ", y_pred.shape)
#    print("x_test shape ", X_test.shape)
#    print("sample y_test ", y_test[19])
#    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
#    accuracy = (tp + tn) / (tp + tn + fp + fn)
#    recall = tp / (tp + fn)
#    precision = tp / (tp + fp)
#    print("accuracy ", accuracy)
#    print("recall ", recall)
#    print("precision ", precision)

