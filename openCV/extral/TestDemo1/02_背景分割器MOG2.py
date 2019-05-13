"""
这是个以高斯混合模型为基础的背景/前景分割算法，这个算法的一个特点是它为每一个像素选择一个合适数目的高斯分布。
这样就会对由于亮度等发生变化引起的场景变化产生更好的适应。首先需要创建一个背景对象。但在这里可以选择是否检测阴影。
如果 detectShadows = True（默认值），它就会检测并将物体标记出来，但是这样做会降低处理速度。
"""
import cv2

cap = cv2.VideoCapture(0)

mog = cv2.createBackgroundSubtractorMOG2()

while(1):
    ret, frame = cap.read()
    fgmask = mog.apply(frame)
    cv2.imshow('frame', fgmask)
    if cv2.waitKey(int(1000 / 12)) & 0xff == ord("q"):
      break

cap.release()
cv2.destroyAllWindows()
