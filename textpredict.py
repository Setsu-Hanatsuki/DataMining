#ライブラリのインポート
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np

#データのロード
category=["talk.religion.misc","soc.religion.christian","sci.space","comp.graphics"]
text=fetch_20newsgroups(remove=("headers","footers","quotes"),categories=category)

#データの前処理
tivec=TfidfVectorizer(max_df=0.9)
x=tivec.fit_transform(text.data)
x=x.toarray()
y=text.target
x=preprocessing.minmax_scale(x)

#データの分割
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

#モデルの生成と学習
clf=clf=MLPClassifier(hidden_layer_sizes=(500,500,250))
clf.fit(x_train, y_train)

#予測
y_pred=clf.predict(x_test)

#精度の検証
print("Accuracy")
print(accuracy_score(y_test,y_pred))
print("Precision")
print(precision_score(y_test, y_pred,average="macro"))
print("Recall")
print(recall_score(y_test, y_pred,average="macro"))
print("F1-score")
print(f1_score(y_test, y_pred,average="macro"))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))

#誤った文章のややこしさ
smvalue=np.array(clf.predict_proba(x_test))
tmp=[]
outvalue=[]
for i in range(len(y_test)):
    if y_test[i]!=y_pred[i]:
        tmp.append(i)
        tmp.append(y_test[i])
        tmp.append(y_pred[i])
        tmp.append(max(smvalue[i]))
        outvalue.append(tmp)
        tmp=[]
print(outvalue)
