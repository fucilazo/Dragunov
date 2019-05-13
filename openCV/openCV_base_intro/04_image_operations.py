import cv2

img = cv2.imread('test.jpg', cv2.IMREAD_COLOR)

px = img[55, 55]    # 获取(55,55)坐标的颜色RGB值
print(px)

roi = img[100:250, 100:250]     # region of image
print(roi)

img[100:250, 100:250] = [255, 255, 255]

pieces = img[37:111, 107:194]
img[0:74, 0:87] = pieces    # 高、宽

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
