import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# CSVァイルを読み込んでデータフレームに格納
df = pd.read_csv("m1-2.csv",encoding="shift-jis")

# 説明変数をダミー変数に変換
x = pd.get_dummies(df[['天気', '風']])

# 目的変数：満足度
y = df['ゴルフ'].values

# 予測モデルを作成(ランダムフォレスト)
clf = RandomForestClassifier(n_estimators=10)
clf.fit(x, y)
importances = clf.feature_importances_

#出力

out=[]
tmp=[]
for i in range(len(x.columns)):
    tmp.append(x.columns[i])
    tmp.append(importances[i])
    out.append(tmp)
    tmp=[]
print(out)
