#ライブラリのインポート
import numpy as np
import librosa
import librosa.display
import os
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from scipy import fftpack

#データのロードと前処理
data=[]
label=[]

for i in range(9):
    path="Asami/1-"+str(i+1)+".mp3"
    a,sr=librosa.load(path)
    a=a[:100000]
    y=librosa.feature.mfcc(y=a,sr=sr)
    data.append(y)
    label.append(0)

for i in range(10):
    path="Yuuki/2-"+str(i+1)+".wav"
    a,sr=librosa.load(path)
    a=a[:100000]
    y=librosa.feature.mfcc(y=a,sr=sr)
    data.append(y)
    label.append(1)
for i in range(len(data)):
    data[i]=sum(data[i])/len(data[i])

#データの分割
train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.3)

#学習
clf=clf=MLPClassifier(hidden_layer_sizes=(500,500,250))
clf.fit(train_x, train_y)

#予測
y_pred=clf.predict(test_x)

#精度の検証
print("Accuracy")
print(accuracy_score(test_y,y_pred))
print("Precision")
print(precision_score(test_y, y_pred,average="macro"))
print("Recall")
print(recall_score(test_y, y_pred,average="macro"))
print("F1-score")
print(f1_score(test_y, y_pred,average="macro"))
