import cv2 as cv

file_name = 'a1.jpeg'


def detection(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #ret, dst = cv.threshold(gray, 200, 255, cv.THRESH_OTSU)
    ret, dst = cv.threshold(gray, 188, 255, cv.THRESH_BINARY_INV)
    return dst


image = cv.imread(file_name)
img = cv.pyrMeanShiftFiltering(src=image, sp=5, sr=40)
dst = detection(img)

src, contours, hierarchy = cv.findContours(dst, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
cv.drawContours(image, contours, -1, (0, 0, 255), 2)
cv.namedWindow('img', cv.WINDOW_NORMAL)
cv.imshow('img', image)
cv.waitKey(0)
