import cv2

vid = cv2.VideoCapture(0)
  
while(True):
    ret, frame = vid.read()
    frame = cv2.flip(frame, 1)
    cv2.imshow('frame', frame)

    if cv2.waitKey(33) == ord('q'):
        print("quit")
        break

vid.release()
cv2.destroyAllWindows()