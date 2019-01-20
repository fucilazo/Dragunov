import cv2

img1 = cv2.imread('test.jpg')
img2 = cv2.imread('test2.jpg')
img3 = cv2.imread('test3.jpg')

add = img1 + img2
add2 = cv2.add(img1, img2)  # add those pixel value together (if >255 equal 255)
weighted = cv2.addWeighted(img1, 0.6, img2, 0.4, 0)

rows, cols, channel = img1.shape
roi = img3[0:rows, 0:cols]

img2gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 220, 255, cv2.THRESH_BINARY_INV)

mask_inv = cv2.bitwise_not(mask)

img3_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
img1_fg = cv2.bitwise_and(img1, img1, mask=mask)

dst = cv2.add(img3_bg, img1_fg)
img3[0:rows, 0:cols] = dst

cv2.imshow('add', add)
cv2.imshow('add2', add2)
cv2.imshow('weighted', weighted)
cv2.imshow('mask', mask)
cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
