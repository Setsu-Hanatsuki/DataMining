#ライブラリのインポート
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.datasets import cifar10
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,classification_report


#データセット(画像)の読み込み
(x_train,y_train),(x_test,y_test)=cifar10.load_data()

#画像データの前処理
x_train, x_test=x_train/255, x_test/255

#CNNの作成
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=x_train.shape[1:]))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(max(y_train)+1, activation='softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


#学習
batch_size = 200 #default=58
num_classes = 10
epochs = 100
earlystop=EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,verbose=1,validation_data=(x_test, y_test),callbacks=[earlystop],shuffle=True)

#予測
y_softmax=model.predict(x_test)
y_pred=[]
for softmax in y_softmax:
    y_pred.append(softmax.argmax())
print("正解率")
print(accuracy_score(y_test,y_pred))
print("再現率")
print(recall_score(y_test,y_pred,average='macro'))
print("適合率")
print(precision_score(y_test,y_pred,average='macro'))
print("F値")
print(f1_score(y_test,y_pred,average='macro'))
print(classification_report(y_test, y_pred))

#誤った画像の保存
for i in range(len(y_pred)):
    if y_test[i]!=y_pred[i]:
        plt.imshow(x_test[i])
        plt.savefig("waim/pred"+str(y_pred[i])+"_test"+str(y_test[i])+"_credit"+str(y_softmax[i][y_softmax[i].argmax()])+".png")
    