import numpy as np
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("-data_dir", help = "dataset folder", type = str)
args = parser.parse_args()
if not args.data_dir:
    print("Dataset directory does not exist")
    exit()
else:
    dataset = Path(args.data_dir)

#pos_x = np.load('pos-train.c.1.1.npy')
#neg_x = np.load('neg-train.c.1.1.npy')
for file in sorted(dataset.glob('*.npy')):
    if 'pos-train' in file.name:
        pos_x = np.load(file)
    if 'neg-train' in file.name:
        neg_x = np.load(file)
    if 'pos-test' in file.name:
        pos_test_x = np.load(file)
    if 'neg-test' in file.name:
        neg_test_x = np.load(file)

pos_y = np.ones((pos_x.shape[0]))
neg_y = np.zeros((neg_x.shape[0]))

pos_test_y = np.ones((pos_test_x.shape[0]))
neg_test_y = np.zeros((neg_test_x.shape[0]))

X = np.vstack((pos_x, neg_x))
print("x shape", X.shape)
y = np.concatenate((pos_y, neg_y), axis = 0)
print("y shape", y.shape)

X_test = np.vstack((pos_test_x, neg_test_x))
print("x test shape", X_test.shape)
y_test = np.concatenate((pos_test_y, neg_test_y), axis = 0)
print("y test shape", y_test.shape)
print("test_postive_samples:",pos_test_y.shape)
print("test_negative_samples:",neg_test_y.shape)

#X=np.vstack((X,X_test))
#y = np.hstack((y.T,y_test.T))
shuffle_idx = np.random.permutation(X.shape[0])
X = X[shuffle_idx]
y = y[shuffle_idx]
shuffle_idx_test = np.random.permutation(X_test.shape[0])
X_test = X_test[shuffle_idx_test]
y_test = y_test[shuffle_idx_test]
#from sklearn.svm import SVC
#from sklearn.datasets import load_iris
#from sklearn.model_selection import StratifiedShuffleSplit
#from sklearn.model_selection import GridSearchCV

#C_range = np.logspace(-2, 10, 13)
#gamma_range = np.logspace(-9, 3, 13)
#param_grid = dict(gamma=gamma_range, C=C_range)
#cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
#grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
#grid.fit(X, y)

#print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))

#from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
from sklearn.metrics import confusion_matrix
#from sklearn.preprocessing import StandardScaler
#X, X_test, y, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
l, m, n = X.shape
X = np.reshape(X, (l, m * n))
a, b, c = X_test.shape
X_test = np.reshape(X_test, (a, b * c))
#print("printing the sizes:",X.shape, X_test.shape)
#scaler = StandardScaler()

#scaler.fit_transform(X)
#scaler.fit(X_test)


clf = svm.SVC()
clf.fit(X, y)
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

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
print("The ROC Score is:",roc_auc_score(y_test, clf.decision_function(X_test)))
metrics.plot_roc_curve(clf, X_test, y_test)
plt.show()

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

