#ライブラリのインポート
import numpy as np
import librosa
import librosa.display
import os
import glob
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import classification_report,confusion_matrix
from scipy import fftpack

#データのロードと前処理
data=[]
label=[]

for i in range(9):
    path="フォルダ名"+str(i+1)+"拡張子名"
    a,sr=librosa.load(path)
    a=a[:100000]#サイズ合わせ
    y=librosa.feature.mfcc(y=a,sr=sr)
    data.append(y)
    label.append(0)

for i in range(10):
    path="フォルダ名"+str(i+1)+"拡張子名"
    a,sr=librosa.load(path)
    a=a[:100000]#サイズ合わせ
    y=librosa.feature.mfcc(y=a,sr=sr)
    data.append(y)
    label.append(1)
for i in range(len(data)):
    data[i]=sum(data[i])/len(data[i])
data = preprocessing.minmax_scale(data)

#データの分割
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.3)

#学習
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
