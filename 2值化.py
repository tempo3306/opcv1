# encoding: utf-8
'''
@author: zhushen
@contact: 810909753@q.com
@time: 2017/7/25 10:38
'''
import numpy as np


import cv2,time
from matplotlib import pyplot as plt
a=time.clock()
img=cv2.imread('86200.png')
print(img)
# print(img[0:10,21:31])


grayimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret,thresh1=cv2.threshold(grayimg,127,255,cv2.THRESH_BINARY_INV)
# image,contours,hierarchy = cv2.findContours(thresh1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
image,contours,hierarchy = cv2.findContours(thresh1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
imgn=[]
print(len(contours))
for i in range(len(contours)):
    cnt=contours[i]
    x,y,w,h=cv2.boundingRect(cnt)
    print(x,y,w,h)
    imgn.append(image[y:y+h,x:x+w])
    # cv2.imshow("Mouth1", imgn[i])
    # cv2.waitKey(0)
    plt.imshow(imgn[i])
    plt.show()

    # plt.imshow()
# plt.show()


####分割
#numpy切片
# roiImg = img[20:120,170:270]



# ret,thresh1=cv2.threshold(grayimg,127,255,cv2.THRESH_BINARY)
# ret,thresh2=cv2.threshold(grayimg,127,255,cv2.THRESH_BINARY_INV)
# ret,thresh3=cv2.threshold(grayimg,127,255,cv2.THRESH_TRUNC)
# ret,thresh4=cv2.threshold(grayimg,127,255,cv2.THRESH_TOZERO)
# ret,thresh5=cv2.threshold(grayimg,127,255,cv2.THRESH_TOZERO_INV)

# b=time.clock()
# print(b-a)
#
#
# titles = ['Gray Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
# images = [grayimg, thresh1, thresh2, thresh3, thresh4, thresh5]
#
#
#
#
# for i in range(6):
#     plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()