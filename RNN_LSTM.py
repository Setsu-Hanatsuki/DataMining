#ライブラリのインポート
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt

#データの読み込み
df=pd.read_csv("1308_2017.csv",encoding="shift-jis")
value=df["終値調整値"]
time2=np.arange(0,len(value)-1,1)

#データの前処理
mx=max(time2)
my=max(value)
time=np.array(time2)
v2=np.array(value)/max(value)
x=[]
y=[]
for i in range(len(v2)-25):
    x.append(v2[i:i+25])
    y.append(v2[i+25])
time2=np.array(x).reshape(len(x),25,1)
value=np.array(y).reshape(len(y),1)

#詳細パラメータ
warnings.filterwarnings('ignore')
future_test=time2[len(time2)-1].T
time_length = future_test.shape[1]
future_result = np.empty((0))
length_of_sequence = time2.shape[1]
in_out_neurons = 1
n_hidden = 300

#LSTM作成
model = Sequential()
model.add(LSTM(n_hidden, batch_input_shape=(None, length_of_sequence, in_out_neurons), return_sequences=False))
model.add(Dense(in_out_neurons))
model.add(Activation('linear'))
optimizer = Adam(lr=1e-3)
model.compile(loss="mean_squared_error", optimizer=optimizer)

#学習
early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=15)
model.fit(time2, value,batch_size=100, epochs=200,validation_split=1.0, callbacks=[early_stopping])

#予測
predicted = model.predict(time2)

#未来のデータの予測
for step in range(50):
    test_data= np.reshape(future_test, (1, time_length, 1))
    batch_predict = model.predict(test_data)
    future_test = np.delete(future_test, 0)
    future_test = np.append(future_test, batch_predict)
    future_result = np.append(future_result, batch_predict)

#可視化
x=[]
y=[]
y3=[]
for i in range(len(predicted)):
    x.append(time[i+20])
    y.append(predicted[i][0])
for i in range(len(future_result)):
    y3.append(future_result[i])
x2 = np.arange(len(predicted)+20-1+4, len(future_result)+len(predicted)+20-1+4)
x=x-(max(x)-max(time))
x2=x2-(min(x2)-max(x))
y2=np.array(y)*my
y4=np.array(y3)*my
plt.plot(x,y2)
plt.plot(x2,y4)
v2=v2*my
plt.plot(v2)
plt.show()
