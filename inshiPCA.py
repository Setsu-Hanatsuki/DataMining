from sklearn.decomposition import PCA
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
pca = PCA(n_components=len(x[0]))
pca.fit(x)
kiyoritsu=pca.explained_variance_ratio_
koyuuvector=pca.components_
out=[]
tmp=[]
for i in range(len(x[0])):
    tmp.append(name[i])
    sump=0
    for j in range(len(koyuuvector)):
        sump=sump+abs(koyuuvector[j][i])*kiyoritsu[j]
    tmp.append(sump)
    out.append(tmp)
    tmp=[]

print(out)
