from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import csv

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
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.5)
model=model=MLPClassifier(hidden_layer_sizes=(500,500,100,))
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
smvalue=np.array(model.predict_proba(X_test))
tmp=[]
outvalue=[]
for i in range(len(y_test)):
    if y_test[i]!=y_pred[i]:
        tmp.append(i)
        tmp.append(y_test[i])
        tmp.append(y_pred[i])
        tmp.append(max(smvalue[i]))
        outvalue.append(tmp)
        tmp=[]
print(outvalue)
