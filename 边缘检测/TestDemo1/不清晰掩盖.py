import cv2
"""
锐化=原始+（原始 - 模糊）×数量
"""
file_name = 'a1.jpeg'
img = cv2.imread(file_name)

img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
# --- smoothing the image ---
blur = cv2.GaussianBlur(img, (23, 23), 0)
# --- applied the formula assuming amount = 1---
img = cv2.add(img[:, :, 1], (img[:, :, 1] - blur[:, :, 1]))

cv2.imshow('sharpened', img)
cv2.waitKey(0)
