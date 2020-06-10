import pandas as pd
from sklearn import linear_model

# CSVファイルを読み込んでデータフレームに格納
df = pd.read_csv("m1.csv",encoding="shift-jis")

# 説明変数をダミー変数に変換
x = pd.get_dummies(df[['天気', '風']])

# 目的変数：満足度
y = df['気温'].values

# 予測モデルを作成(重回帰)
reg = linear_model.LinearRegression()
reg.fit(x, y)

# 回帰係数と切片の抽出
a = reg.coef_
b = reg.intercept_  

#出力

out=[]
tmp=[]
for i in range(len(x.columns)):
    tmp.append(x.columns[i])
    tmp.append(a[i])
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
