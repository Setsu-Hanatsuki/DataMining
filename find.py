from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

image=np.array(Image.open("Face.jpg"))

#顔検出
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
faces=face_cascade.detectMultiScale(gray,1.3,5)
for (x,y,w,h) in faces:
    img=cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)
    roi_gray=gray[y:y+h,x:x+w]
    roi_color=img[y:y+h,x:x+w]
plt.imshow(img)
plt.show()

plt.imshow(roi_color)
plt.show()
