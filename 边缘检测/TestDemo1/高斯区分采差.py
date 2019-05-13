import cv2

file_name = 'a1.jpeg'


def nothing():
    pass


cv2.namedWindow('image')
cv2.createTrackbar('sigmaX_1', 'image', 0, 15, nothing)
cv2.createTrackbar('sigmaY_1', 'image', 0, 15, nothing)
cv2.createTrackbar('sigmaX_2', 'image', 0, 15, nothing)
cv2.createTrackbar('sigmaY_2', 'image', 0, 15, nothing)
img = cv2.imread(file_name)
img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

while True:
    sigmaX_1 = cv2.getTrackbarPos('sigmaX_1', 'image')  # 1
    sigmaY_1 = cv2.getTrackbarPos('sigmaY_1', 'image')  # 2
    sigmaX_2 = cv2.getTrackbarPos('sigmaX_2', 'image')  # 4
    sigmaY_2 = cv2.getTrackbarPos('sigmaY_2', 'image')  # 7

    blur1 = cv2.GaussianBlur(img, (5, 5), sigmaX_1, sigmaY_1)
    blur2 = cv2.GaussianBlur(img, (5, 5), sigmaX_2, sigmaY_2)
    DoG_edge = blur1 - blur2

    cv2.imshow('image', DoG_edge)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
