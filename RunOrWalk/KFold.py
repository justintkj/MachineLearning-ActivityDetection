import sklearn
from sklearn.model_selection import KFold
from sklearn import metrics
# import metrics we'll need
from sklearn.metrics import accuracy_score  
import numpy as np
import timeit

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
    trainer.fit(X_list, y_list)
    sample_data = X_list[0:1, 0:]
    start_time = timeit.default_timer()
    pred_one = trainer.predict(sample_data)
    total_time = (timeit.default_timer() - start_time)*1000
    print ("Time taken for one prediction: {0:.50f}".format(total_time))
    return final_accuracy/ count
