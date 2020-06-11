import csv
import numpy as np
from sklearn import linear_model

f=open("rent.csv","r")
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
                tmp.append(float(row[i]))
            else:
                y.append(float(row[i]))
        x.append(tmp)
        tmp=[]
reg = linear_model.LinearRegression()
reg.fit(x, y)
x=np.array(x)
y_pred=reg.predict(x)
sse=np.sum((y-y_pred)**2,axis=0)
sse=sse/(x.shape[0]-x.shape[1]-1)
s=np.linalg.inv(np.dot(x.T,x))
std_err=np.sqrt(np.diagonal(sse*s))

a = reg.coef_
b = reg.intercept_
out=[]
tmp=[]
for i in range(len(name)):
    tmp.append(name[i])
    tmp.append(a[i])
    tmp.append(a[i]/std_err[i])
    out.append(tmp)
    tmp=[]
tmp.append("切片")
tmp.append(b)
out.append(tmp)
tmp=[]
tmp.append("決定係数")
tmp.append(reg.score(x,y))
out.append(tmp)
print(out)
