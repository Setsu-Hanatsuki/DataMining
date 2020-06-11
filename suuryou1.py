import pandas as pd
from sklearn import linear_model

df = pd.read_csv("m1.csv",encoding="shift-jis")

x = pd.get_dummies(df[['天気', '風']])

y = df['気温'].values

reg = linear_model.LinearRegression()
reg.fit(x, y)

a = reg.coef_
b = reg.intercept_  

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
