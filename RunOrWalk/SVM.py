##### imports
import sklearn
from sklearn import metrics
def SVMprocess():
    # import metrics we'll need
    from sklearn.metrics import accuracy_score  
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve 
    from sklearn.metrics import auc
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
        
    ##Support Vector Machine
    from sklearn.svm import SVC
    from nb_author_id import preprocesses
    from KFold import KFoldalgo

    X_list, y_list = preprocesses()
    # instantiate time
    import time
    start_time = time.time()
    # instantiate the estimator
    svm = SVC()

    # fit the model
    #svm.fit(X_train, y_train)
    kfold_acc = KFoldalgo(X_list,y_list,svm)
    pred_svm = kfold_acc
    # predict the response
    #y_pred = svm.predict(X_test)

    # accuracy score
    #pred_svm = metrics.accuracy_score(y_test, y_pred)
    print ("Accuracy for SVM: {}".format(pred_svm))
    print ("Time taken for SVM: {}".format(time.time()-start_time))

    return time.time()-start_time, pred_svm
