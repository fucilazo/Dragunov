"""
均值漂移（Meanshift）是一种目标跟踪算法，该算法寻找概率函数离散样本的最大密度，并重新计算在下一帧中的最大密度，给出了目标的移动方向。
"""
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('surveillance_demo/768x576.avi')
# capture the first frame
ret,frame = cap.read()
# mark the ROI
r,h,c,w = 10, 200, 10, 200
# wrap in a tuple
track_window = (c,r,w,h)

# extract the ROI for tracking
roi = frame[r:r+h, c:c+w]# 提取感兴趣区域ROI
# switch to HSV
hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)# 转换为HSV色彩空间
# create a mask with upper and lower boundaries of colors you want to track
# # 下面是创建一个包含具有HSV值的ROI所有像素的掩码，HSV值在上界与下界之间
mask = cv2.inRange(hsv_roi, np.array((100., 30.,32.)), np.array((180.,120.,255.)))
# calculate histograms of roi
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])# 计算图像的色彩直方图
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)# 计算直方图后，响应的值被归一化到0-255范围内。

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )# 指定均值漂移终止一系列计算行为的方式
# 这里的停止条件为：均值漂移迭代10次后或者中心移动至少1个像素时，均值漂移就停止计算中心漂移
# 第一个标志（EPS或CRITERIA_COUNT）表示将使用两个条件的任意一个（计数或‘epsilon’，意味着哪个条件最先达到就停止）

while(1):
    ret ,frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        # print dst
        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Draw it on image
        x,y,w,h = track_window
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv2.imshow('img2',img2)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break

    else:
        break

cv2.destroyAllWindows()
cap.release()
