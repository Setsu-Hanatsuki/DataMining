from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt


image=np.array(Image.open("noise.jpg"))

#plt.imshow(image)
#plt.show()

#輪郭強調
kernel=np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]],np.float32)
dst=cv2.filter2D(image,-1,kernel)
dst=image+dst
#plt.imshow(dst)
#plt.show()

#コントラスト強調
alpha=1.0
beta=30.0
con=image*alpha+beta
con=np.clip(con,0,255).astype(np.uint8)
#plt.imshow(con)
#plt.show()

#ノイズフィルタリング(非線形)
fil=cv2.medianBlur(image,9)
#plt.imshow(fil)
#plt.show()

con2=fil*alpha+beta
con2=np.clip(con,0,255).astype(np.uint8)
con2=cv2.medianBlur(image,5)
plt.imshow(con2)
plt.show()
