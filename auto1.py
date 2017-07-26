# encoding: utf-8
'''
@author: zhushen
@contact: 810909753@q.com
@time: 2017/7/26 15:36
'''
import pyautogui as pg

image=pg.screenshot("1.png",region=(800,800,20,20))

import time

a=time.clock()
qa=pg.locateOnScreen(image)

print(qa)
b=time.clock()
print(b-a)