#!/usr/bin/python
import numpy as np
import pandas as pd

def preprocesses():
    df=pd.read_csv('../input/dataset.csv', na_values = "?")
    df.dropna(inplace=True)
    df.index = np.arange(1, len(df)+1)
    df.corr()
    df1=df.drop(['date','time','username','wrist'],axis=1)
    df1

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

    data = df1.values
    X = data[:, 1:]  
    y = data[:, 0]  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

