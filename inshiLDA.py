from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import preprocessing
import numpy as np
import pandas as pd

#データの読み込み
df=pd.read_csv("wine.csv")
y_table=df["Wine"]
x_table=df.drop(columns="Wine")
y=y_table.values
x=x_table.values
name=x_table.columns

#データの前処理
x = preprocessing.minmax_scale(x)

#モデルの定義
model=LDA(n_components=len(x[9]),store_covariance=True)

#学習
model.fit(x,y)

#因子重要度
imp=model.coef_

#出力
out=[]
out.append(name)
for i in range(len(imp)):
    out.append(imp[i])
out=np.array(out).T
dfo=pd.DataFrame(out)
dfo.columns=["項目","重要度クラス1","重要度クラス2","重要度クラス3"]
print(dfo)
