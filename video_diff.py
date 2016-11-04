import cv2
import numpy as np

cap = cv2.VideoCapture('vid1.mp4.wmv')

ret, old = cap.read()

while(cap.isOpened()):
    ret, frame = cap.read()

    cv2.imshow('difference', cv2.absdiff(frame, old))
    cv2.imshow('original', frame)

    old = frame
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
