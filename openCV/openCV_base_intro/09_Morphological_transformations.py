import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()   # '_' means it is no use in this part
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_color = np.array([0, 0, 0])
    upper_color = np.array([100, 255, 200])

    mask = cv2.inRange(hsv, lower_color, upper_color)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(mask, kernel, iterations=1)
    dilation = cv2.dilate(mask, kernel, iterations=1)

    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('frame', frame)
    cv2.imshow('res', res)
    cv2.imshow('erosion', erosion)
    cv2.imshow('dilation', dilation)
    cv2.imshow('opening', opening)
    cv2.imshow('closing', closing)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
