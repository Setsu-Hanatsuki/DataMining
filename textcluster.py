#ライブラリのインポート
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd
import matplotlib.pyplot as plt


#データのロード
category=["talk.religion.misc","soc.religion.christian","sci.space","comp.graphics"]
text=fetch_20newsgroups(remove=("headers","footers","quotes"),categories=category)

#データの前処理
tivec=TfidfVectorizer(max_df=0.9)
x=tivec.fit_transform(text.data)
x=x.toarray()
y=text.target

#デンドログラム作成
pca=PCA(n_components=100)
pca.fit(x)
tpc=pca.transform(x)
df=pd.DataFrame(tpc)
Z=linkage(df,method="median",metric="euclidean")
dendrogram(Z,labels=y)
plt.savefig("dendrogram.pdf")
plt.show()
