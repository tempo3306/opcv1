# encoding: utf-8
'''
@author: zhushen
@contact: 810909753@q.com
@time: 2017/7/25 10:38
'''
import cv2,time
from matplotlib import pyplot as plt
a=time.clock()
img=cv2.imread('1.jpg')
print(img)

grayimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)



ret,thresh1=cv2.threshold(grayimg,127,255,cv2.THRESH_BINARY)
ret,thresh2=cv2.threshold(grayimg,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3=cv2.threshold(grayimg,127,255,cv2.THRESH_TRUNC)
ret,thresh4=cv2.threshold(grayimg,127,255,cv2.THRESH_TOZERO)
ret,thresh5=cv2.threshold(grayimg,127,255,cv2.THRESH_TOZERO_INV)

b=time.clock()
print(b-a)


titles = ['Gray Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [grayimg, thresh1, thresh2, thresh3, thresh4, thresh5]




for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()