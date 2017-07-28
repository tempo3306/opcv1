# encoding: utf-8
'''
@author: zhushen
@contact: 810909753@q.com
@time: 2017/7/27 17:10
'''
""" 
Created on Thu Mar 23 15:46:59 2017 

@author: onlyyo 
批量将切割后并且已经分好类的图像，得到的图片进行二值化处理，变成像素值，然后保存在TXT文件下 
"""
from PIL import Image
import numpy as np
import os


# 特征提取，获取图像二值化数学值
def getBinaryPix(im):
    im = Image.open(im)
    img = np.array(im)
    rows, cols = img.shape
    for i in range(rows):
        for j in range(cols):
            if (img[i, j] <= 128):
                img[i, j] = 0
            else:
                img[i, j] = 1

    binpix = np.ravel(img)
    return binpix


def getfiles(dirs):
    fs = []
    for fr in os.listdir(dirs):
        f = dirs + fr
        if f.rfind(u'.DS_Store') == -1:
            fs.append(f)
    return fs


def writeFile(content):
    with open(u'J:/数据分析学习/python/机器学习之验证码识别/traindata/train_data.txt', 'a+') as f:
        f.write(content)
        f.write('\n')
        f.close()

import cv2
import numpy as np
SZ=20
bin_n = 16 # Number of bins
# svm_params = dict( kernel_type = cv2.SVM_LINEAR,svm_type = cv2.SVM_C_SVC,C=2.67, gamma=5.383 )
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n * ang / (2 * np.pi))  # quantizing binvalues in (0...16)
    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)  # hist is a 64 bit vector
    return hist

a=cv2.imread("screen.png")


# if __name__ == '__main__':
#     dirs = u'J:/数据分析学习/python/机器学习之验证码识别/category/%s/'
#
#     for i in range(9):
#         for f in getfiles(dirs % (i)):
#             pixs = getBinaryPix(f).tolist()
#             pixs.append(i)
#             pixs = [str(i) for i in pixs]
#             content = ','.join(pixs)
#             writeFile(content)