import cv2
import numpy as np

img = cv2.pyrDown(cv2.imread("lighting.png", cv2.IMREAD_UNCHANGED))
ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    # find bounding box coordinates
    # 现计算出一个简单的边界框
    x, y, w, h = cv2.boundingRect(c)   # 将轮廓信息转换成(x, y)坐标，并加上矩形的高度和宽度
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)  # 画出矩形

    # find minimum area
    # 计算包围目标的最小矩形区域
    rect = cv2.minAreaRect(c)
    # calculate coordinate of the minimum area rectangle
    box = cv2.boxPoints(rect)
    # normalize coordinates to integers
    box =np.int0(box)
    # 注：OpenCV没有函数能直接从轮廓信息中计算出最小矩形顶点的坐标。所以需要计算出最小矩形区域，
    # 然后计算这个矩形的顶点。由于计算出来的顶点坐标是浮点型，但是所得像素的坐标值是整数（不能获取像素的一部分），
    # 所以需要做一个转换
    # draw contours
    cv2.drawContours(img, [box], 0, (0, 0, 255), 3)  # 画出该矩形

    # calculate center and radius of minimum enclosing circle
    (x, y), radius = cv2.minEnclosingCircle(c)  # 会返回一个二元组，第一个元素为圆心的坐标组成的元组，第二个元素为圆的半径值。
    # cast to integers
    center = (int(x), int(y))
    radius = int(radius)
    # draw the circle
    img = cv2.circle(img, center, radius, (0, 255, 0), 2)

cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
cv2.imshow("contours", img)
cv2.waitKey()
cv2.destroyAllWindows()
