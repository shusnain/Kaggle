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

from sklearn import datasets, neighbors, linear_model

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

n_samples = len(X_digits)
size = n_samples * .9
X_train = X_digits[:n_samples]
y_train = y_digits[:n_samples]
X_test = X_digits[n_samples:]
y_test = y_digits[n_samples:]

knn = neighbors.KNeighborsClassifier()
logistic = linear_model.LogisticRegression()

y_pred_knn = knn.fit(X_train, y_train).predict(X_test)
y_pred_logistic = logistic.fit(X_train, y_train).predict(X_test)
data_knn=[]
data_logistic=[]
for i in range(n_samples):
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
