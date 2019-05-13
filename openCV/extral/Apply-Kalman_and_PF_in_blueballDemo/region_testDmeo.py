import cv2
import numpy as np


if __name__ == '__main__':
    def is_color(region):
        # limit the blur color range
        return (region >= 90) & (region < 130)

    img = cv2.imread('test.jpg')
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_h = img_hsv[:, :, 0]
    img_h_mask = img_h.copy()
    img_h_mask2 = img_h.copy()
    img_s = img_hsv[:, :, 1]
    img_v = img_hsv[:, :, 2]
    _, img_h2 = cv2.threshold(img_hsv[:, :, 0], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, img_s2 = cv2.threshold(img_hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, img_v2 = cv2.threshold(img_hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_h_mask[(img_s2 == 0) | (img_v2 == 0)] = 0
    # img_h_mask2[(img_s2 == 0) | (img_v2 == 0)] = 0
    # region = img_h_mask[130:340, 220:460]
    region = img_h_mask[230:340, 320:460]
    count = region[is_color(region)].size

    # cv2.imshow('img', img)
    # cv2.imshow('hsv', img_hsv)
    # cv2.imshow('h', img_h)
    # cv2.imshow('s', img_s)
    # cv2.imshow('v', img_v)
    # cv2.imshow('h2', img_h2)
    # cv2.imshow('s2', img_s2)
    # cv2.imshow('v2', img_v2)
    #
    # cv2.imshow('mask', img_h_mask)
    cv2.imshow('region', region)
    # print(region.shape[0])
    # print(region.shape[1])

    # print(is_color(region))
    # for i in range(region.shape[1]):
    #     print(region[0][i])

    # print(region.size)
    # print(count)    # 25211

    cv2.waitKey(0)
    cv2.destroyAllWindows()
