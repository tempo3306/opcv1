# encoding: utf-8
'''
@author: zhushen
@contact: 810909753@q.com
@time: 2017/7/27 8:07
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt



img = cv2.imread("screen.png",0)  #截屏
img2 = img.copy()
template=cv2.imread("target.png",0)   #要寻找对象的对象
w,h=template.shape[::-1]

res=cv2.matchTemplate(img,template, cv2.TM_CCOEFF_NORMED)
min_val,max_val,min_loc,max_loc=cv2.minMaxLoc(res)
print(min_val)
print(max_val)
print(min_loc)
print(max_loc)












