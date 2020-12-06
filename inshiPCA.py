#ライブラリのインポート
from sklearn.decomposition import PCA
from sklearn import preprocessing
import numpy as np
import pandas as pd

#データの読み込み
df=pd.read_csv("wine.csv")
x_table=df.drop(columns="Wine")
x=x_table.values
name=x_table.columns

#データの前処理
x = preprocessing.minmax_scale(x)

#モデルの定義
pca = PCA(n_components=len(x[0]))

#学習
pca.fit(x)

#寄与率
con=pca.explained_variance_ratio_

#固有ベクトル
fac=pca.components_

#因子負荷量
tmp=[]
total=[]
for i in range(len(con)):
    for j in range(len(fac[i])):
        tmp.append(np.sqrt(con[i])*fac[i][j])
    total.append(tmp)
    tmp=[]
out=[]
total=np.array(total).T
for i in range(len(total)):
    tmp.append(name[i])
    tmp.append(sum(abs(total[i])))
    out.append(tmp)
    tmp=[]
dfo=pd.DataFrame(out)
dfo.columns=["因子名","因子負荷量総計"]
print(dfo)
