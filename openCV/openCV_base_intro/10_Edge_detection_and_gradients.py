import numpy as np
import cv2

cap = cv2.VideoCapture(0)
img = cv2.imread('test.jpg', cv2.IMREAD_COLOR)
img2 = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)

laplaction2 = cv2.Laplacian(img, cv2.CV_64F)
laplaction3 = cv2.Laplacian(img2, cv2.CV_64F)

cv2.imshow('laplaction2', laplaction2)

# while True:
#     _, frame = cap.read()   # '_' means it is no use in this part
#
#     laplaction = cv2.Laplacian(frame, cv2.CV_64F)
#
#     cv2.imshow('original', frame)
#
#     k = cv2.waitKey(5) & 0xFF
#     if k == 27:
#         break

cv2.destroyAllWindows()
cap.release()
