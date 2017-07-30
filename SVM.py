# encoding: utf-8
'''
@author: zhushen
@contact: 810909753@q.com
@time: 2017/7/26 12:36
'''
# 训练的点数
import matplotlib.pyplot as plt
import numpy as np
import cv2

train_pts = 30

# 创建测试的数据点，2类
# 以(-1.5, -1.5)为中心
rand1 = np.ones((train_pts, 2)) * (-2) + np.random.rand(train_pts, 2)
print('rand1：')
print(rand1)

# 以(1.5, 1.5)为中心
rand2 = np.ones((train_pts, 2)) + np.random.rand(train_pts, 2)
print('rand2:')
print(rand2)

# 合并随机点，得到训练数据
train_data = np.vstack((rand1, rand2))
train_data = np.array(train_data, dtype='float32')
train_label = np.vstack((np.zeros((train_pts, 1), dtype='int32'), np.ones((train_pts, 1), dtype='int32')))

# 显示训练数据
# plt.figure(1)
# plt.plot(rand1[:, 0], rand1[:, 1], 'o')
# plt.plot(rand2[:, 0], rand2[:, 1], 'o')
# plt.plot(rand2[:, 0], rand2[:, 1], 'o')
# plt.show()

# 创建分类器
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)  # SVM类型
svm.setKernel(cv2.ml.SVM_LINEAR) # 使用线性核
svm.setC(1)


# 训练
ret = svm.train(train_data, cv2.ml.ROW_SAMPLE, train_label)
print("fsdfd")
print(train_data.shape)
print(train_label)
# 测试数据，20个点[-2,2]
pt = np.array(np.random.rand(20,2) * 4 - 2, dtype='float32')
(ret, res) = svm.predict(pt)
print("res = ")
print(res)

# 按label进行分类显示
plt.figure(2)
res = np.hstack((res, res))

# 第一类
type_data = pt[res < 0.5]
# print(type_data)
type_data = np.reshape(type_data, (int(type_data.shape[0] / 2), 2))
# print(type_data)
plt.plot(type_data[:, 0], type_data[:, 1], 'o')
# 第二类
type_data = pt[res >= 0.5]
type_data = np.reshape(type_data, (int(type_data.shape[0] / 2), 2))
plt.plot(type_data[:, 0], type_data[:, 1], 'o')

# plt.show()

vec = svm.getSupportVectors()
print("df",vec)
print("dfs")




