import cv2

img = cv2.imread('test_mid.jpg', cv2.IMREAD_COLOR)
img1 = img[0:int(img.shape[0]/2), 0:]
img2 = img1[0:, 0:int(img1.shape[1]/2)]
j = 0
k = 0
cv2.imshow('image', img)
# print(img.shape)    # (450, 720, 3)

# for h in range(img.shape[0]):
#     for l in range(img.shape[1]):
#         # 逐行
#         if l % 2 == 0:
#             img1[h, l1] = img[h, l]
#             l1 += 1

for i in range(img.shape[0]):
    if i % 2 == 0:
        pieces = img[i:i+1, 0:]
        img1[j:j+1, 0:] = pieces
        j += 1
cv2.imshow('image1', img1)

for i in range(img1.shape[1]):
    if i % 2 == 0:
        pieces = img1[0:, i:i+1]
        img2[0:, k:k+1] = pieces
        k += 1
cv2.imshow('image2', img2)

cv2.waitKey(0)
cv2.destroyAllWindows()
