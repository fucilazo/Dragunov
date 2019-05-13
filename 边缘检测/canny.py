import cv2
import numpy as np

file_name = 'carscale_0252.jpg'
img = cv2.imread(file_name)
cv2.imshow('img', img)
"""
cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient ]]])   
必要参数：
第一个参数是需要处理的原图像；
第二个参数是滞后阈值1；
第三个参数是滞后阈值2。
"""
cv2.imshow('img2', cv2.Canny(img, 100, 300))
cv2.waitKey(0)
