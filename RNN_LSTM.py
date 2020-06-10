#In[1]
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import csv

f=open('kabu.csv','r',encoding='utf-8')
reader=csv.reader(f)
time2=[]
gdp2=[]
jikeiretu=[]
for row in reader:
    jikeiretu.append(float(row[0]))
    time2.append(float(row[0]))
    gdp2.append(float(row[2]))

Mt=max(time2)
Mg=max(gdp2)
time=np.array(time2)
time3=np.arange(1994,2021,0.25)
gdp=np.array(gdp2)/max(gdp2)
x=[]
y=[]
for i in range(len(gdp)-25):
    x.append(gdp[i:i+25])
    y.append(gdp[i+25])
time2=np.array(x).reshape(len(x),25,1)
gdp2=np.array(y).reshape(len(y),1)
#In[2]
warnings.filterwarnings('ignore')


#In[6]

#In[7]

#In[8]
future_test=time2[len(time2)-1].T

#In[9]
time_length = future_test.shape[1]

#In[10]
future_result = np.empty((0))

#In[11]
length_of_sequence = time2.shape[1]
in_out_neurons = 1
n_hidden = 300

#In[12]-Model Definision-
model = Sequential()
model.add(LSTM(n_hidden, batch_input_shape=(None, length_of_sequence, in_out_neurons), return_sequences=False))
model.add(Dense(in_out_neurons))
model.add(Activation('linear'))
optimizer = Adam(lr=1e-3)
model.compile(loss="mean_squared_error", optimizer=optimizer)

#In[13]-Learning-
early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=15)

#In[14]
model.fit(time2, gdp2,batch_size=100, epochs=100,validation_split=0.1, callbacks=[early_stopping])

#In[15]-Prediction | Training data-
predicted = model.predict(time2)

#In[16]-Prediction | Future data-
for step in range(50):
    test_data= np.reshape(future_test, (1, time_length, 1))
    batch_predict = model.predict(test_data)
    
    future_test = np.delete(future_test, 0)
    future_test = np.append(future_test, batch_predict)
    
    future_result = np.append(future_result, batch_predict)

x=[]
y=[]
y3=[]
for i in range(len(predicted)):
#for i in range(len(future_result)):
    x.append(time[i+20])
    #x.append(i+20)
    y.append(predicted[i][0])
#print(time2)
for i in range(len(future_result)):
    y3.append(future_result[i])
#print(future_result)
#x2 = np.arange(0+len(function), len(future_result)+len(function))
x2 = np.arange(len(predicted)+20-1, len(future_result)+len(predicted)+20-1)
y2=np.array(y)*Mg
y4=np.array(y3)*Mg
plt.plot(x,y2)
plt.plot(x2,y4)
gdp=gdp*Mg
x=np.arange(0,len(gdp),1)
plt.plot(time,gdp)
plt.show()
