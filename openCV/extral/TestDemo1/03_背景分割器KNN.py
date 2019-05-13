import cv2

bs =  cv2.createBackgroundSubtractorKNN(detectShadows=True)
# camera = cv2.VideoCapture("shipin.flv")
camera = cv2.VideoCapture(0)

while(1):
    ret, frame = camera.read()
    fgmask = bs.apply(frame)# 计算获得前景掩码
    th = cv2.threshold(fgmask.copy(), 244, 255, cv2.THRESH_BINARY)[1]# 设定阈值
    dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=2)# 识别目标
    contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv2.contourArea(c) > 1600:
            (x, y, w, h) = cv2.boundingRect(c)# 检测轮廓
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,0), 2)# 绘制检测结果

    cv2.imshow('mog', fgmask)
    cv2.imshow('thresh', th)
    cv2.imshow('detection', frame)

    if cv2.waitKey(int(1000 / 12)) & 0xff == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
