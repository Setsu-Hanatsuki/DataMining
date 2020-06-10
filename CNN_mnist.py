import os
os.environ["KERAS_BACKEND"] = "tensorflow"
kerasBKED = os.environ["KERAS_BACKEND"] 
print(kerasBKED)

import keras
from keras.models import load_model
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import pickle
import numpy as np

batch_size = 58
num_classes = 10
epochs = 100

from PIL import Image
import numpy as np
import os
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
data=load_digits()
x=data.images
y=data.target
X=[]
K1=[]
K2=[]
K3=[]
X2=[]
print(x.shape)
plt.imshow(x[2])
plt.show()
for i in range(len(x)):
    minx=x[i][0][0]
    maxx=x[i][0][0]
    for j in range(len(x[i])):
        for k in range(len(x[i][j])):
            #for l in range(len(x[i][j][k])):
            K1.append(int(x[i][j][k]))
            K1.append(int(x[i][j][k]))
            K1.append(int(x[i][j][k]))
            if x[i][j][k] < minx:
                minx=x[i][j][k]
            if x[i][j][k] > maxx:
                maxx=x[i][j][k]
            K2.append(K1)
            K1=[]
        K3.append(K2)
        K2=[]
    X.append(K3)
    K3=[]
    for j in range(len(x[i])):
        for k in range(len(x[i][j])):
            for l in range(len(X[i][j][k])):
                X[i][j][k][l]=int((X[i][j][k][l]-minx)*255/(maxx-minx))
X3=np.array(X)
X3.reshape(1797, 8, 8, 3)
#for i in range(len(X3[0][0])):
#    print(X3[0][0][i])
plt.imshow(X[2])
plt.show()
x_train=X3
y_train=y

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.3)

#print(len(Y_test))
#print(Y_test)

Y_traincl = keras.utils.to_categorical(Y_train, num_classes)
Y_testcl = keras.utils.to_categorical(Y_test, num_classes)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

#print(X_train[0][0][0])

featureLayer1=[Conv2D(64, (3, 3), padding='same',input_shape=X_train.shape[1:]),
               Activation('relu'),
               Conv2D(64, (3, 3), padding='same'),
               Activation('relu'),
               MaxPooling2D(pool_size=(2, 2)),
               Dropout(0.25)]

featureLayer2=[Conv2D(128, (3, 3), padding='same'),
               Activation('relu'),
               Conv2D(128, (3, 3), padding='same'),
               Activation('relu'),
               MaxPooling2D(pool_size=(2, 2)),
               Dropout(0.25)]

featureLayer3=[Conv2D(256, (3, 3), padding='same'),
               Activation('relu'),
               Conv2D(256, (3, 3), padding='same'),
               Activation('relu'),
               Conv2D(256, (3, 3), padding='same'),
               Activation('relu'),
               MaxPooling2D(pool_size=(2, 2)),
               Dropout(0.25)]

fullConnLayer=[Flatten(),
               Dense(1024),
               Activation('relu'),
               Dropout(0.5),
               Dense(1024),
               Activation('relu'),
               Dropout(0.5)]

classificationLayer=[Dense(num_classes),
                     Activation('softmax')]

model = Sequential(featureLayer1 + featureLayer2 + featureLayer3 + fullConnLayer + classificationLayer)

opt = keras.optimizers.adam()

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

es_cb = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')

history = model.fit(X_train, Y_traincl,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, Y_testcl),
                    callbacks=[es_cb],
                    shuffle=True)

from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()
import pandas as pd
Y_testP = []
Y_testPred = model.predict(X_test)
for x in Y_testPred:
    Y_testP.append(x.argmax())
Y_testP = np.array(Y_testP)
labels = sorted(list(set(Y_test)))
cmx_data = confusion_matrix(Y_test, Y_testP, labels=labels)
df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
plt.figure(figsize = (10,7))
sns.heatmap(df_cmx, annot=True)
plt.show()

SM=[]
for i in range(len(Y_testPred)):
    for j in range(len(Y_testPred[i])):
        SM.append(max(Y_testPred[i]))
        
plt.boxplot(SM)
plt.show()
