#ライブラリのインポート
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np

#データのロード
category=["talk.religion.misc","soc.religion.christian","sci.space","comp.graphics"]
text=fetch_20newsgroups(remove=("headers","footers","quotes"),categories=category)

#データの前処理
tivec=TfidfVectorizer(max_df=0.9)
x=tivec.fit_transform(text.data)
x=x.toarray()
y=text.target

#次元圧縮
pca = PCA(n_components=2)
pca.fit(x)
tx=pca.transform(x)

#図示
tx=preprocessing.minmax_scale(tx)
model=model=MLPClassifier(hidden_layer_sizes=(500,500,250))
model.fit(tx,y)
fig,ax=plt.subplots(figsize=(8,6))
X, Y = np.meshgrid(np.linspace(*ax.get_xlim(), 1000), np.linspace(*ax.get_ylim(), 1000))
XY = np.column_stack([X.ravel(), Y.ravel()])
Z =model.predict(XY).reshape(X.shape)
plt.contourf(X, Y, Z, alpha=0.1, cmap='brg')
plt.scatter(tx[:, 0], tx[:, 1], c=y, s=50, cmap='brg')
plt.xlim(min(tx[:,0]),max(tx[:,0]))
plt.ylim(min(tx[:,1]),max(tx[:,1]))
plt.show()
