#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：BadouCV 
@File    ：toushiAPI.py
@IDE     ：PyCharm 
@Author  ：zjd
@Date    ：2024/11/1 19:36
'''
import cv2
import numpy as np

img = cv2.imread('./IMG_2861.JPG')

result3 = img.copy()

'''
注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
'''
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
print(img.shape)
# 生成透视变换矩阵；进行透视变换
m = cv2.getPerspectiveTransform(src, dst)
print("warpMatrix:")
print(m)
result = cv2.warpPerspective(result3, m, (337, 488))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)