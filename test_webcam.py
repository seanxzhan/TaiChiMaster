import cv2

cam_idx = 2

vid = cv2.VideoCapture(cam_idx)
  
while(True):
    ret, frame = vid.read()
    if cam_idx == 0:
        frame = cv2.flip(frame, 1)
    cv2.imshow('frame', frame)

    if cv2.waitKey(33) == ord('q'):
        print("quit")
        break

vid.release()
cv2.destroyAllWindows()