import cv2 as cv

file_name = 'a1.jpeg'


def enhance(img):
    img = cv.resize(img, (0, 0), fx = 0.3, fy = 0.3)
    blur = cv.GaussianBlur(img, (23, 23), 0)
    img = cv.add(img[:, :, 1], (img[:, :, 1] - blur[:, :, 1]))
    return img


def detection(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #ret, dst = cv.threshold(gray, 200, 255, cv.THRESH_OTSU)
    ret, dst = cv.threshold(gray, 188, 255, cv.THRESH_BINARY_INV)
    return dst


image = cv.imread(file_name)
img = enhance(image)
dst = detection(img)

src, contours, hierarchy = cv.findContours(dst, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
cv.drawContours(img, contours, -1, (0, 0, 255), 2)
cv.namedWindow('img', cv.WINDOW_NORMAL)
cv.imshow('img', img)
cv.waitKey(0)