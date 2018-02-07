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
def RFprocess():
    X_train, X_test, y_train, y_test = preprocesses()
    import time
    start_time = time.time()
    # instantiate the estimator
    clf = RandomForestClassifier(random_state=1)

    # fit the model
    clf.fit(X_train, y_train)

    # predict the response
    y_pred = clf.predict(X_test)

    # accuracy score
    pred_rf = metrics.accuracy_score(y_test, y_pred)
    print ("Accuracy for RandomForest: {}".format(pred_rf))
    print ("Time taken for RandomForest: {}".format(time.time()-start_time))

    return time.time()-start_time, pred_rf

