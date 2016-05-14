"""
================================
Digits Classification Exercise
================================

A tutorial exercise regarding the use of classification techniques on
the Digits dataset.

This exercise is used in the :ref:`clf_tut` part of the
:ref:`supervised_learning_tut` section of the
:ref:`stat_learn_tut_index`.
"""
print(__doc__)

from sklearn import neighbors, linear_model
import numpy as np
import pandas as pd
import os
import csv

if __name__ == '__main__':
 path = "D:/Dropbox/Data Science/Kaggle/Scikit-learn"
 os.chdir(path)
 train_data = pd.read_csv(path + '/data/train.csv', header=None)
 train_labels = pd.read_csv(path + '/data/trainLabels.csv', header=None)
 test_data = pd.read_csv(path + '/data/test.csv', header=None)
 X_train = np.asarray(train_data)
 Y_train = np.asarray(train_labels).ravel()
 X_test= np.asarray(test_data)


knn = neighbors.KNeighborsClassifier()
logistic = linear_model.LogisticRegression()

y_pred_knn = knn.fit(X_train, Y_train).predict(X_test)
y_pred_logistic = logistic.fit(X_train, Y_train).predict(X_test)
data_knn=[]
data_logistic=[]
for i in range(9000):
  data_knn.append([i+1,y_pred_knn[i]])
  data_logistic.append([i+1,y_pred_logistic[i]])

with open('knn.csv', 'w', newline='') as csvfile:
    out = csv.writer(csvfile, delimiter=',')
    out.writerow(['Id','Solution'])
    out.writerows(data_knn)

with open('logistic.csv', 'w', newline='') as csvfile:
    out = csv.writer(csvfile, delimiter=',')
    out.writerow(['Id','Solution'])
    out.writerows(data_logistic)