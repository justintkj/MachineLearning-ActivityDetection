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
    
##Random Forest
from sklearn.ensemble import RandomForestClassifier
from nb_author_id import preprocesses
from KFold import KFoldalgo
from loo import looalgo
from ConfusionMatrix import confusionMatrixAlgo 

def RFprocess():
    X_list, y_list = preprocesses()
    import time
    start_time = time.time()
    # instantiate the estimator
    rndforest = RandomForestClassifier(random_state=1)
    kfold_acc = KFoldalgo(X_list,y_list,rndforest)
    end_time = time.time()                          #considers the run-time only while using KFold and not loo
    pred_rf_kfold = kfold_acc
    loo_acc = looalgo(X_list, y_list, rndforest)
    pred_rf_loo = loo_acc

    # Confusion Matrix
    rndforest.fit(X_list, y_list)
    y_pred = rndforest.predict(X_list)
    con_matrix = confusionMatrixAlgo(y_list, y_pred)
    # fit the model
    #clf.fit(X_train, y_train)

    # predict the response
    #y_pred = clf.predict(X_test)

    # accuracy score
    #pred_rf = metrics.accuracy_score(y_test, y_pred)
    print ("Accuracy for RandomForest Using KFold Cross Validation: {}".format(pred_rf_kfold))
    print ("Accuracy for RandomForest Using Leave One Out Cross Validation: {}".format(pred_rf_loo))
    #print ("Time taken for RandomForest: {}".format(end_time-start_time))

    return time.time()-start_time, pred_rf_kfold, pred_rf_loo
