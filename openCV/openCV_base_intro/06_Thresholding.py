import cv2

img = cv2.imread('shadow.jpg')
retval, threshold = cv2.threshold(img, 9, 255, cv2.THRESH_BINARY)

grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
retval2, threshold2 = cv2.threshold(grayscale, 9, 255, cv2.THRESH_BINARY)
gaus = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
retval2, otsu = cv2.threshold(grayscale, 125, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imshow('img', img)
cv2.imshow('threshold', threshold)
cv2.imshow('threshold2', threshold2)    # It is simply convert color to gray scale, but not BRG
cv2.imshow('gaus', gaus)
cv2.imshow('otsu', otsu)
cv2.waitKey(0)
cv2.destroyAllWindows()