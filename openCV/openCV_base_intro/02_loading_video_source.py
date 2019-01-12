import cv2

cap = cv2.VideoCapture(0)   # 参数为0为打开笔记本内置摄像头，参数为路径为打开指定路径视频
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out.write(frame)
    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray)

    if cv2.waitKey(1) and 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()


