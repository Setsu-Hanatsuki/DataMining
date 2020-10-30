#ライブラリのインポート
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import csv

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
clf=DecisionTreeClassifier()
clf.fit(x, y)

#単語の重要度測定
name=tivec.get_feature_names()
imp=clf.feature_importances_
total=[]
total.append(name)
total.append(imp)

#保存
fb = open("importance_word.csv", 'w',encoding='utf-8')
writer = csv.writer(fb, lineterminator='\n')
writer.writerows(total)
fb.close()
