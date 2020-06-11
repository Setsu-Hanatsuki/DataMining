import pandas as pd
from sklearn.decomposition import PCA

df = pd.read_csv("m1.csv",encoding="shift-jis")

x = pd.get_dummies(df[['天気', '風']])

pca = PCA(n_components=len(x.columns))
pca.fit(x)
kiyoritsu=pca.explained_variance_ratio_
koyuuvector=pca.components_
out=[]
tmp=[]

for i in range(len(x.columns)):
    tmp.append(x.columns[i])
    sump=0
    for j in range(len(koyuuvector)):
        sump=sump+abs(koyuuvector[j][i])*kiyoritsu[j]
    tmp.append(sump)
    out.append(tmp)
    tmp=[]

print(out)
