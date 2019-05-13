"""
在你移动的过程中你会发现蓝色框的大小是固定的，如果由远及近的运动的话，固定的大小是不合适的。
所以我们需要根据目标的大小和角度来对窗口的大小和角度进行修订。这就需要使用CAMShift技术了。
"""
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# take first frame of the video
ret,frame = cap.read()

# setup initial location of window
r,h,c,w = 10,200,10,200  # simply hardcoded the values
track_window = (c,r,w,h)


roi = frame[r:r+h, c:c+w]# 提取感兴趣区域ROI
hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)# 转换为HSV色彩空间
# 下面是创建一个包含具有HSV值的ROI所有像素的掩码，HSV值在上界与下界之间
mask = cv2.inRange(hsv_roi, np.array((100., 30.,32.)), np.array((180.,120.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])# 计算图像的色彩直方图
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)# 计算直方图后，响应的值被归一化到0-255范围内。
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )# 指定均值漂移终止一系列计算行为的方式
# 这里的停止条件为：均值漂移迭代10次后或者中心移动至少1个像素时，均值漂移就停止计算中心漂移
# 第一个标志（EPS或CRITERIA_COUNT）表示将使用两个条件的任意一个（计数或‘epsilon’，意味着哪个条件最先达到就停止）

while(1):
    ret ,frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)# 直方图反向投影 得到一个矩阵（每个像素以概率的形式表示）

        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        pts = cv2.boxPoints(ret)# 找到被旋转矩形的顶点，而折线函数会在帧上绘制矩形的线段
        pts = np.int0(pts)
        img2 = cv2.polylines(frame,[pts],True, 255,2)

        cv2.imshow('img2',img2)
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break

    else:
        break

cv2.destroyAllWindows()
cap.release()
