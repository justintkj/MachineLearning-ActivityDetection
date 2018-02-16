##### imports
import sklearn
from sklearn import metrics
# import metrics we'll need
from sklearn.metrics import accuracy_score  
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve 
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
    
##Naive Bayes
from sklearn.naive_bayes import GaussianNB
from nb_author_id import preprocesses
from KFold import KFoldalgo

import time
def NBprocess():
    start_time = time.time()
    X_list, y_list = preprocesses()
    # instantiate the estimator
    nb = GaussianNB()

    # fit the model
    kfold_acc = KFoldalgo(X_list,y_list,nb)
    pred_nb = kfold_acc
    
    # predict the response
    #y_pred = nb.predict(X_test)

    # accuracy score
    #pred_nb = metrics.accuracy_score(y_test, y_pred)
    print ("Accuracy for Gaussian Naive Bayes: {}".format(pred_nb))
    print ("Time taken for Naive Bayes: {}".format(time.time()-start_time))


    return time.time()-start_time, pred_nb



