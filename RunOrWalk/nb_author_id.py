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
    from math import ceil    
    data = df1.values
    total_sample = len(data)
    portion = 1
    #Sample data are from 2nd column onwards
    X = data[0:int(total_sample*portion), 1:]
    #Labels are given only in the first column
    y = data[0:int(total_sample*portion), 0]
    #segment the data by window size and overlap
    window_size = 50;
    overlap = 25;
    segment = []
    labels = []
    for i in range(len(X)/overlap):
        segment.append(X[i*overlap:((i*overlap)+(window_size)), 0:])        
        labels.append(y[i*overlap:((i*overlap)+(window_size))])    
    stat_list = []
    label_list = []
    #extract features
    for i in range(len(segment)):
        temp_row = []
        for j in range(0,6):
            temp = segment[i][0:,j]
            #Mean = sum of everything / no. of data point
            mean = np.mean(temp)
            #Median = middle value of sorted
            median = np.median(temp)
            #Std = Standard Deviation, How individual points differs from the mean
            std = np.std(temp)
            #iqr = Inter-Quartile Range, 75th percentile - 25th percentile
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
            #Fourier = Power Spectral Density, essentially, Summation |Ck|^2
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
            #if majority is "1", it means running, (2)
            label_list.append(int(2))
        elif(mean >=0.333):
            #if half is "1", half is "0", jogging (1)
            label_list.append(int(1))
        else:
            #if majority is "0", resting (0)
            label_list.append(int(0))
    #normalizing the features
    stat_list = preprocessing.normalize(stat_list)
    #use this code to extract top 5 features only
    #stat_list = stat_list[0: ,[5,7,10,11,35]]
    
    #X_train, X_test, y_train, y_test = train_test_split(stat_list, label_list,train_size=0.75,test_size=0.25, random_state=42)
    #scaler = StandardScaler()
    #scaler.fit(X_train)
    #X_train = scaler.transform(X_train)
    #X_test = scaler.transform(X_test)
    return stat_list, label_list

def main():
    preprocesses()

main()

