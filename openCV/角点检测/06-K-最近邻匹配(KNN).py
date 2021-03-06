import numpy as np
import cv2
from matplotlib import pyplot as plt

# 读取图片内容
img1 = cv2.imread('aa.png',0)
img2 = cv2.imread('bb.png',0)

# 使用ORB特征检测器和描述符，计算关键点和描述符
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# 暴力匹配BFMatcher，遍历描述符，确定描述符是否匹配，然后计算匹配距离并排序
# BFMatcher函数参数：
# normType：NORM_L1, NORM_L2, NORM_HAMMING, NORM_HAMMING2。
# NORM_L1和NORM_L2是SIFT和SURF描述符的优先选择，NORM_HAMMING和NORM_HAMMING2是用于ORB算法
bf = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
# knnMatch 函数参数k是返回符合匹配的个数，暴力匹配match只返回最佳匹配结果。
matches = bf.knnMatch(des1,des2,k=1)

# 使用plt将两个图像的第一个匹配结果显示出来
# 若使用knnMatch进行匹配，则需要使用drawMatchesKnn函数将结果显示
img3 = cv2.drawMatchesKnn(img1=img1,keypoints1=kp1,img2=img2,keypoints2=kp2, matches1to2=matches, outImg=img2, flags=2)
plt.imshow(img3)
plt.show()
