import numpy as np
import cv2
from matplotlib import pyplot as plt

# 读取图片内容
img1 = cv2.imread('aa.png', 0)
img2 = cv2.imread('bb.png', 0)
# img1 = cv2.imread('girl.jpg', 0)
# img2 = cv2.imread('misaka_mikoto.jpg', 0)

# 使用ORB特征检测器和描述符，计算关键点和描述符
orb = cv2.xfeatures2d.SURF_create(float(4000))
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)


# 暴力匹配BFMatcher，遍历描述符，确定描述符是否匹配，然后计算匹配距离并排序
# BFMatcher函数参数：
# normType：NORM_L1, NORM_L2, NORM_HAMMING, NORM_HAMMING2。
# NORM_L1和NORM_L2是SIFT和SURF描述符的优先选择，NORM_HAMMING和NORM_HAMMING2是用于ORB算法
bf = cv2.BFMatcher(normType=cv2.NORM_L1, crossCheck=True)
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)
# matches是DMatch对象，具有以下属性：
# DMatch.distance - 描述符之间的距离。 越低越好。
# DMatch.trainIdx - 训练描述符中描述符的索引
# DMatch.queryIdx - 查询描述符中描述符的索引
# DMatch.imgIdx - 训练图像的索引。

# 获取aa.jpg的关键点位置
x,y = kp1[matches[0].queryIdx].pt
cv2.rectangle(img1, (int(x),int(y)), (int(x) + 5, int(y) + 5), (0, 255, 0), 2)
cv2.imshow('a', img1)

# 获取bb.png的关键点位置
x1,y1 = kp2[matches[0].trainIdx].pt
cv2.rectangle(img2, (int(x1),int(y1)), (int(x1) + 5, int(y1) + 5), (0, 255, 0), 2)
cv2.imshow('b', img2)

# 使用plt将两个图像的匹配结果显示出来
img3 = cv2.drawMatches(img1=img1,keypoints1=kp1,img2=img2,keypoints2=kp2, matches1to2=matches[:1], outImg=img2, flags=2)
plt.imshow(img3)
plt.show()
