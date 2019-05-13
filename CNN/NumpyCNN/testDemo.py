import skimage
import matplotlib.pyplot as plt
import cv2

img = skimage.data.chelsea()
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()