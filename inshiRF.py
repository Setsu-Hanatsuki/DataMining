#ライブラリのインポート
from sklearn.ensemble import RandomForestClassifier
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
model=RandomForestClassifier(n_estimators=10)

#学習
model.fit(x,y)

#因子重要度
imp=model.feature_importances_

#出力
out=[]
out.append(name)
out.append(imp)
out=np.array(out).T
dfo=pd.DataFrame(out)
dfo.columns=["項目","重要度"]
print(dfo)
