#!/usr/bin/python
import numpy as np
import pandas as pd
import scipy as sc

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
    from scipy.fftpack import fft
    # import metrics we'll need
    from sklearn.metrics import accuracy_score  
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve 
    from sklearn.metrics import auc
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn import preprocessing
    
    data = df1.values
    X = data[:, 1:]
    y = data[:, 0]
    #segment the data by window size and overlap
    window_size = 50;
    overlap = 25;
    segment = []
    labels = []
    for i in range(len(data)/overlap):
        segment.append(X[i*overlap:((i*overlap)+(window_size)), 0:])        
        labels.append(y[i*overlap:((i*overlap)+(window_size))])    
    stat_list = []
    label_list = []
    #extract features
    for i in range(len(segment)):
        temp_row = []
        for j in range(0,6):
            temp = segment[i][0:,j]
            mean = np.mean(temp)
            median = np.median(temp)
            std = np.std(temp)
            iqr = np.percentile(temp, [75 ,25])
            q75, q25 = np.percentile(temp, [75 ,25])
            iqr = q75 - q25
            maximum = np.amax(temp)
            temp_row.append(mean)
            temp_row.append(median)
            temp_row.append(std)
            temp_row.append(iqr)
            temp_row.append(maximum)
            Fourier_temp = fft(temp)
            fourier = np.abs(Fourier_temp)**2
            value = 0
            for x in range (len(fourier)):
                value = value + (fourier[x] * fourier [x])
            value = value / len(fourier)
            temp_row.append(value)
        stat_list.append(temp_row)

    #extract label
    for i in range(len(labels)):
        mean = np.mean(labels[i])
        if(mean >=0.666):
            label_list.append(int(2))
        elif(mean >=0.333):
            label_list.append(int(1))
        else:
            label_list.append(int(0))
    #normalizing the features
    stat_list = preprocessing.normalize(stat_list)
    #X_train, X_test, y_train, y_test = train_test_split(stat_list, label_list,train_size=0.75,test_size=0.25, random_state=42)
    #scaler = StandardScaler()
    #scaler.fit(X_train)
    #X_train = scaler.transform(X_train)
    #X_test = scaler.transform(X_test)
    return stat_list, label_list

def main():
    preprocesses()

main()

