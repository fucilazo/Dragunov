import cv2
import numpy as np

camera = cv2.VideoCapture(0)

es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
kernel = np.ones((5,5),np.uint8)
background = None

while (True):
  ret, frame = camera.read()
  if background is None:  # 第一帧作为背景输入
    background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    # 先将帧转换为灰阶
    background = cv2.GaussianBlur(background, (21, 21), 0)  # 再进行一次模糊处理（平滑）
    continue

  gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      # 先将帧转换为灰阶
  gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)    # 再进行一次模糊处理（平滑）
  diff = cv2.absdiff(background, gray_frame)                # 计算与背景的差异，并得到一个差分图
  diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1] # 应用阈值得到一副黑白图，
  diff = cv2.dilate(diff, es, iterations = 2)               # 膨胀(dilate)图像，从而对孔(hole)和缺陷(imperfection)进行归一化处理
  cnts, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)# 计算一幅图中目标的轮廓

  for c in cnts:
    if cv2.contourArea(c) < 1500:
      continue
    (x, y, w, h) = cv2.boundingRect(c)  # 计算矩形的边界框
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

  cv2.imshow("contours", frame)
  cv2.imshow("dif", diff)
  if cv2.waitKey(int(1000 / 12)) & 0xff == ord("q"):
      break

cv2.destroyAllWindows()
camera.release()
