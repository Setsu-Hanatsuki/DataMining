from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image=np.array(Image.open("noise.jpg"))

for i in range(len(image)):
    for j in range(len(image[i])):
        tmp1=[]
        tmp2=[]
        tmp3=[]
        try:
            tmp1.append(image[i-1][j-1][0])
            tmp2.append(image[i-1][j-1][1])
            tmp3.append(image[i-1][j-1][2])
            
        except:
            a=1
        try:
            tmp1.append(image[i][j-1][0])
            tmp2.append(image[i][j-1][1])
            tmp3.append(image[i][j-1][2])
            
        except:
            a=1
        try:
            tmp1.append(image[i+1][j-1][0])
            tmp2.append(image[i+1][j-1][1])
            tmp3.append(image[i+1][j-1][2])
            
        except:
            a=1
        try:
            tmp1.append(image[i-1][j][0])
            tmp2.append(image[i-1][j][1])
            tmp3.append(image[i-1][j][2])
            
        except:
            a=1
        try:
            tmp1.append(image[i][j][0])
            tmp2.append(image[i][j][1])
            tmp3.append(image[i][j][2])
            
        except:
            a=1
        try:
            tmp1.append(image[i+1][j][0])
            tmp2.append(image[i+1][j][1])
            tmp3.append(image[i+1][j][2])
            
        except:
            a=1
        try:
            tmp1.append(image[i-1][j+1][0])
            tmp2.append(image[i-1][j+1][1])
            tmp3.append(image[i-1][j+1][2])
            
        except:
            a=1
        try:
            tmp1.append(image[i][j+1][0])
            tmp2.append(image[i][j+1][1])
            tmp3.append(image[i][j+1][2])
            
        except:
            a=1
        try:
            tmp1.append(image[i+1][j+1][0])
            tmp2.append(image[i+1][j+1][1])
            tmp3.append(image[i+1][j+1][2])
            
        except:
            a=1
        tmp1=np.array(tmp1)
        tmp2=np.array(tmp2)
        tmp3=np.array(tmp3)
        image[i][j][0]=np.median(tmp1)
        image[i][j][1]=np.median(tmp2)
        image[i][j][2]=np.median(tmp3)
plt.imshow(image)
plt.show()
