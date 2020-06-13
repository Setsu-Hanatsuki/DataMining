from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import numpy as np
import csv
import matplotlib.pyplot as plt
f=open("wine.csv","r")
reader=csv.reader(f)
k=0
name=[]
tmp=[]
x=[]
y=[]
for row in reader:
    if k==0:
        for j in range(len(row)):
            if j!=0:
                name.append(row[j])
        k=1
    else:
        for i in range(len(row)):
            if i!=0:
                tmp.append(row[i])
            else:
                y.append(int(row[i]))
        x.append(tmp)
        tmp=[]
x = preprocessing.minmax_scale(x)
model=model=MLPClassifier(hidden_layer_sizes=(500,500,100,))
model.fit(x,y)
kfold = KFold(n_splits=int(len(x)/10))#10%
scores=cross_val_score(model, x, y, cv=kfold)
#print(scores)
num=np.arange(1,len(scores)+1,1)
plt.bar(num,scores)
plt.xlabel("Data Group")
plt.ylabel("accuracy score")
plt.show()
