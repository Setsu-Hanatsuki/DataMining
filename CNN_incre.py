from tensorflow.keras import layers,models
from keras.layers import Input
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from PIL import Image, ImageOps
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

#黒縁
def preprocess(img):
    h, w, c = img.shape
    longest_edge = max(h, w)
    top = 0
    bottom = 0
    left = 0
    right = 0
    if h < longest_edge:
        diff_h = longest_edge - h
        top = diff_h // 2
        bottom = diff_h - top
    elif w < longest_edge:
        diff_w = longest_edge - w
        left = diff_w // 2
        right = diff_w - left
    else:
        pass
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img=cv2.resize(img,(int(longest_edge*0.25),int(longest_edge*0.25)))
    return img

#水増し
def incimg(image):
    src=np.array(image)
    r90=np.array(image.rotate(90))
    r18=np.array(image.rotate(180))
    r27=np.array(image.rotate(270))
    flp=np.array(ImageOps.flip(image))
    mrr=np.array(ImageOps.mirror(image))
    x1=np.hstack((src,r90))
    x1=np.hstack((x1,r18))
    x2=np.hstack((r27,flp))
    x2=np.hstack((x2,mrr))
    nimg=np.vstack((x1,x2))
    nimg=nimg.astype("float32")
    nimg/=255
    nimg=cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    return nimg


#データの読み込み
x=[]
y=[]
for path in glob.glob("./2/*"):
    image=Image.open(path)
    nimg=incimg(image)
    nimg=preprocess(nimg)
    x.append(nimg)
    y.append(0)
    
for path in glob.glob("./4/*"):
    image=Image.open(path)
    nimg=incimg(image)
    nimg=preprocess(nimg)
    x.append(nimg)
    y.append(1)

for path in glob.glob("./10/*"):
    image=Image.open(path)
    nimg=incimg(image)
    nimg=preprocess(nimg)
    x.append(nimg)
    y.append(2)

#データの前処理
x=np.array(x)

#データの分割
#x,x_test,y,y_test=train_test_split(x,y,test_size=0.2)
#x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=0.25)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


#モデルの宣言
model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=x_train.shape[1:]))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(max(y_train)+1, activation='softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


#モデルの学習
batch_size = 58 #default=58
num_classes = 10
epochs = 150
earlystop=EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
"""
history=model.fit(x_train,
          y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_val, y_val),
          callbacks=[earlystop],
          shuffle=True,
          steps_per_epoch=len(x_train))
"""
history=model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,verbose=1,validation_data=(x_test, y_test),callbacks=[earlystop],shuffle=True)


#モデルの予測
y_predSM=model.predict(x_test)
y_pred=[]
for sm in y_predSM:
    y_pred.append(sm.argmax())

#精度の検証
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))

#学習曲線
acc=history.history["acc"]
val_acc=history.history["val_acc"]
loss=history.history["loss"]
val_los=history.history["val_loss"]
ep=range(len(acc))
plt.plot(ep,acc,label="acc")
plt.plot(ep,val_acc,label="val_acc")
plt.legend()
plt.show()
plt.plot(ep,loss,label="loss")
plt.plot(ep,val_loss,label="val_loss")
plt.legend()
plt.show()
"""
#ややこしい画像
badimage=[]
tmp=[]
for i in range(len(y_pred)):
    if y_pred[i]!=y_test[i]:
        tmp.append(i)
        tmp.append(y_test[i])
        tmp.append(y_pred[i])
        tmp.append(max(y_predSM[i]))
        badimage.append(tmp)
        tmp=[]
"""
