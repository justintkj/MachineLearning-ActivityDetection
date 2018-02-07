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
X_train, X_test, y_train, y_test = preprocesses()
#Create plot
g1 = list()
g2 = list()
g3 = list()
g4 = list()
g5 = list()
g6 = list()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax = fig.add_subplot(1,1,1,axisbg= "1.0")
for x in range(0, len(X_train)):
    if y_train[x] == 0:
        g1.append(X_train[x][0])
    else:
        g2.append(X_train[x][0])
for x in range(0, len(X_train)):
    if y_train[x] == 0:
        g3.append(X_train[x][1])
        g5.append(X_train[x][2])
    else:
        g4.append(X_train[x][1])
        g6.append(X_train[x][2])
data = (g1,g2)
data2 = (g3,g4)
data3 = (g5,g6)
colors = ("red","green","blue")
groups = ("walk", "run")
for data,data2,data3, color, group in zip(data,data2,data3, colors, groups):
    xs = data
    ys = data2
    zs = data3
    #ax.scatter(x, y alpha=0.8, c=color, edgecolors='none', s=30, label=group)
    ax.scatter(xs, ys, zs, zdir='z', s=20, c=None, depthshade=True)

plt.title('Matplot scatter plot')
plt.legend(loc=2)
plt.show()


