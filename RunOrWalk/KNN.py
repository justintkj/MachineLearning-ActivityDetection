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
from sklearn.neighbors import KNeighborsClassifier
from nb_author_id import preprocesses
from KFold import KFoldalgo
def KNNprocess():
    X_list, y_list = preprocesses()
    import time
    start_time = time.time()
    # instantiate the estimator
    knn = KNeighborsClassifier()
    kfold_acc = KFoldalgo(X_list, y_list, knn)
    pred_knn = kfold_acc
    # fit the model
    #knn.fit(X_train, y_train)

    # predict the response
    #y_pred = knn.predict(X_test)

    # accuracy score
    #pred_knn = metrics.accuracy_score(y_test, y_pred)
    print ("Accuracy for Knn: {}".format(pred_knn))
    print ("Time taken for knn: {}".format(time.time()-start_time))

    return time.time()-start_time, pred_knn
