import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# CSVァイルを読み込んでデータフレームに格納
df = pd.read_csv("m1.csv",encoding="shift-jis")

# 説明変数をダミー変数に変換
x = pd.get_dummies(df[['天気', '風','ゴルフ']])

#主成分分析で圧縮
pca = PCA(n_components=2)
pca.fit(x)
tx=pca.transform(x)

#図示
for i in range(len(tx)):
    plt.text(tx[i][0],tx[i][1],str(i))
plt.scatter(tx[:,0],tx[:,1])
    
plt.xlim(min(tx[:,0])-0.5,max(tx[:,0])+0.5)
plt.ylim(min(tx[:,1])-0.5,max(tx[:,1])+0.5)
plt.show()

