import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("m1-2.csv",encoding="shift-jis")

x = pd.get_dummies(df[['天気', '風']])

y = df['ゴルフ'].values

clf = RandomForestClassifier(n_estimators=10)
clf.fit(x, y)
importances = clf.feature_importances_

out=[]
tmp=[]
for i in range(len(x.columns)):
    tmp.append(x.columns[i])
    tmp.append(importances[i])
    out.append(tmp)
    tmp=[]
print(out)
