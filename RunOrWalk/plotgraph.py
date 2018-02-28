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
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nb_author_id import preprocesses
X_train,y_train = preprocesses()
#Create plot
g1 = list()
g2 = list()
g3 = list()
g4 = list()
g5 = list()
g6 = list()
g7 = list()
g8 = list()
g9 = list()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax = fig.add_subplot(1,1,1,axisbg= "1.0")
for x in range(0, len(X_train)):
    if y_train[x] == 0:
        g1.append(X_train[x][11])
        g4.append(X_train[x][10])
        g7.append(X_train[x][5])
    elif y_train[x] == 1:
        g2.append(X_train[x][11])
        g5.append(X_train[x][10])
        g8.append(X_train[x][5])
    else:
        g3.append(X_train[x][11])
        g6.append(X_train[x][10])
        g9.append(X_train[x][5])
        
data = (g1,g2,g3)
data2 = (g4,g5,g6)
data3 = (g7,g8,g9)
colors = ("red","green","blue")
groups = ("rest","walk", "run")
for data,data2,data3, color, group in zip(data,data2,data3, colors, groups):
    xs = data
    ys = data2
    zs = data3
    #ax.scatter(x, y alpha=0.8, c=color, edgecolors='none', s=30, label=group)
    ax.scatter(xs, ys, zs, zdir='z', s=20, c=color,label = group, depthshade=True)

plt.title('Scatter plot for Run/Walk')
plt.legend(loc=2)
ax.set_xlabel('ESD,accX')
ax.set_ylabel('Max,accY')
ax.set_zlabel('ESD,accY')
plt.show()


