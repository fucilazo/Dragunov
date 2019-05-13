import cv2

file_name = 'carscale_0252.jpg'


def nothing(x):
    pass


cv2.namedWindow('image')
cv2.createTrackbar('min', 'image', 0, 500, nothing)
cv2.createTrackbar('max', 'image', 0, 500, nothing)
img = cv2.imread(file_name)
img = cv2.GaussianBlur(img, (3, 3), 0)
while 1:
    a1 = cv2.getTrackbarPos('min', 'image')
    a2 = cv2.getTrackbarPos('max', 'image')
    edges = cv2.Canny(img, a1, a2)
    cv2.imshow('image', edges)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
