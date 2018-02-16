import sklearn
from sklearn.model_selection import KFold
from sklearn import metrics
# import metrics we'll need
from sklearn.metrics import accuracy_score  
import numpy as np

def KFoldalgo(X_list, y_list, trainer):
    kf = KFold(n_splits = 50)
    kf.get_n_splits(X_list)
    count = 0
    final_accuracy = 0
    for train_index, test_index in kf.split(X_list):
        X_train, X_test = X_list[train_index], X_list[test_index]
        y_list = np.asarray(y_list)
        y_train, y_test = y_list[train_index], y_list[test_index]
        trainer.fit(X_train, y_train)
        y_pred = trainer.predict(X_test)
        pred_val = metrics.accuracy_score(y_test, y_pred)
        final_accuracy = final_accuracy + pred_val
        count = count +1
    return final_accuracy/ count
