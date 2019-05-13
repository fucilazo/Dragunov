import cv2
import numpy as np

img = cv2.imread("test.jpg")
cv2.imshow('1', img)
print(np.mean(img, axis=(0, 1)))

img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
cv2.imshow('2', img)
print(np.mean(img, axis=(0, 1)))

cv2.waitKey(0)
cv2.destroyAllWindows()
