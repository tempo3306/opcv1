# encoding: utf-8
'''
@author: zhushen
@contact: 810909753@q.com
@time: 2017/7/28 15:52
'''
import cv2
import numpy as np
SZ=20
bin_n = 16 # Number of bins

####可以将斜着的数字摆正
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img
#hog特征
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


#二值化，切割
def cut(img):

    ret,thresh1=cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
    # image,contours,hierarchy = cv2.findContours(thresh1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    image,contours,hierarchy = cv2.findContours(thresh1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    imgn=[]
    xy=[]
    for i in range(len(contours)):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        print(x, y, w, h)
        xy.append([x, y, w, h])

    xy = sorted(xy)
    for i in range(len(contours)):
        x, y, w, h = xy[i]
        imgn.append(image[y:y + h, x:x + w])
    return imgn
img1=cv2.imread("10.png",0)
a=hog(img1)
#SVM训练
#创建训练素材
# 合并随机点，得到训练数据
img1=cv2.imread("10.png",0)
trainData=cut(img1)
trainData=list(map(hog,trainData))
# trainData=np.array(trainData,dtype='float32')
trainData = np.float32(trainData).reshape(-1,bin_n*4)
print(trainData.shape)
# trainData = trainData.reshape(-1,64)
responses=np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[0]],dtype='int32')
print(responses)
# 创建分类器
# svm_params = dict( kernel_type = cv2.ml.SVM_LINEAR,
#                        svm_type = cv2.ml.SVM_C_SVC,
#                        C = 1  )
# #训练
# print(len(trainData))
# svm = cv2.ml.SVM_create()
# ret=svm.train(trainData,responses, params=svm_params)    #responses标签，trainData输入的训练集
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)  # SVM类型
svm.setKernel(cv2.ml.SVM_LINEAR) # 使用线性核
svm.setC(1)


# 训练
ret = svm.train(trainData, cv2.ml.ROW_SAMPLE, responses)
# img2=cv2.imread("86200.png",0)
img2=cv2.imread("90100.png",0)
testData=cut(img2)
testData=list(map(hog,testData))
# trainData=np.array(trainData,dtype='float32')
testData = np.float32(testData).reshape(-1,bin_n*4)
result = svm.predict(testData)
svm.save('svm_cat_data.dat')
# SVM_load('svm_cat_data.dat')
print(result)