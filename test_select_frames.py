import cv2
import numpy as np
from select_frames import *
from time import sleep

#filename = 'vid1.mp4'
#corners = np.array([[10, 150], [10, 462], [348, 468], [350, 146]],
#                   dtype=np.int32)


filename = 'tests/go game.mp4'
corners = np.array([[7, 79], [7, 403], [354, 409], [354, 80]],
                         dtype=np.int32)


cap = FrameSelector(filename)
cap.initialize()

ret, frame = cap.read()
cv2.imshow('video', frame)
cv2.imshow('last good', cap.last_good_frame)

board_mask = np.zeros(frame.shape[0:2], dtype=np.uint8)
board_mask = cv2.fillConvexPoly(board_mask, corners[:,::-1], 1)
cap.set_roi(None, board_mask)

try:
    while cap.isOpened():
        ret, frame = cap.check()
        cv2.imshow('video', frame)
        cv2.imshow('diff', cap.diff*255)
        if ret:
            cv2.imshow('last good', frame)

        if cv2.waitKey(1) > 0:
            break
except:
    cap.release()
    cv2.destroyAllWindows()
    raise

cap.release()
cv2.destroyAllWindows()
