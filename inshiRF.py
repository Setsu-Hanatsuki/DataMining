from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
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
clf = RandomForestClassifier(n_estimators=10)
clf.fit(x, y)
importances = clf.feature_importances_
out=[]
tmp=[]
for i in range(len(x[0])):
    tmp.append(name[i])
    tmp.append(importances[i])
    out.append(tmp)
    tmp=[]
print(out)
