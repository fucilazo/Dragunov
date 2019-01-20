import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()   # '_' means it is no use in this part.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_color = np.array([0, 0, 0])
    upper_color = np.array([100, 255, 200])

    mask = cv2.inRange(hsv, lower_color, upper_color)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    kernel = np.ones((15, 15), np.float32)/225
    smoothed = cv2.filter2D(res, -1, kernel)

    blur = cv2.GaussianBlur(res, (15, 15), 0)

    median = cv2.medianBlur(res, 15)

    bilateral = cv2.bilateralFilter(res, 15, 75, 75)

    cv2.imshow('frame', frame)
    # cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    cv2.imshow('smoothed', smoothed)
    cv2.imshow('blur', blur)    # It is looks more clear.
    cv2.imshow('median', median)    # Contain less noisy.
    cv2.imshow('bilateral', bilateral)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
