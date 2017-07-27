# encoding: utf-8
'''
@author: zhushen
@contact: 810909753@q.com
@time: 2017/7/26 15:36
'''
import pyautogui as pg
import time
time.sleep(3)

image=pg.screenshot("screen.png")
image2=pg.screenshot("target.png",region=(400,400,60,20))



import time

a=time.clock()
qa=pg.locateOnScreen(image)

print(qa)
b=time.clock()
print(b-a)