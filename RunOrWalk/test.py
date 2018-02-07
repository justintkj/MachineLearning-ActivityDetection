from sklearn.model_selection import LeaveOneOut
import numpy as np
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([1, 2, 1])
loo = LeaveOneOut()
loo.get_n_splits(X)
print(loo)
LeaveOneOut()
for train_index, test_index in loo.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(X_train, X_test, y_train, y_test)
