import matplotlib.pyplot as plt
import cv2
#                               0
img = cv2.imread('mid.jpg', cv2.IMREAD_GRAYSCALE)
# IMREAD_COLOR = 1
# IMREAD_UNCHANGED = -1
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imwrite('watchgray,png', img)

plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.show()
