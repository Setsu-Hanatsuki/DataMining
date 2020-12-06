#ライブラリのインポート
import pandas as pd
import numpy as np
from sklearn import linear_model

#データの読み込み
df=pd.read_csv("rent.csv",encoding="shift-jis")
y_table=df["家賃"]
x_table=df.drop(columns="家賃")
y=y_table.values
x=x_table.values
name=x_table.columns

#モデルの定義
model=linear_model.LinearRegression()

#学習
model.fit(x,y)

#モデル
a = model.coef_     #重み
b = model.intercept_#切片

#予測値の生成
y_pred=model.predict(x)

#標準誤差を算出
se=np.sum((y-y_pred)**2,axis=0)
se=se/(x.shape[0]-x.shape[1]-1)
s=np.linalg.inv(np.dot(x.T,x))
stder=np.sqrt(np.diagonal(se*s))

#t値を算出
t=a/stder

#出力
out=[]
out.append(name)
out.append(a)
out.append(t)
out=np.array(out).T
dfo=pd.DataFrame(out)
dfo.columns=["因子名","重み","t値"]
print(dfo)
print("決定係数\t"+str(model.score(x,y)))
print("切片\t"+str(b))
