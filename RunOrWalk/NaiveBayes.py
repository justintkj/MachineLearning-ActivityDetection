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

X_train, X_test, y_train, y_test = preprocesses()
# instantiate the estimator
nb = GaussianNB()

# fit the model
nb.fit(X_train, y_train)

# predict the response
y_pred = nb.predict(X_test)

# accuracy score
pred_nb = metrics.accuracy_score(y_test, y_pred)
print ("Accuracy for Gaussian Naive Bayes: {}".format(pred_nb))