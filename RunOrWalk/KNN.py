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

X_train, X_test, y_train, y_test = preprocesses()
# instantiate the estimator
knn = KNeighborsClassifier()

# fit the model
knn.fit(X_train, y_train)

# predict the response
y_pred = knn.predict(X_test)

# accuracy score
pred_knn = metrics.accuracy_score(y_test, y_pred)
print ("Accuracy for Knn: {}".format(pred_knn))
