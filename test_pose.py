import numpy as np
import cv2
import pose

##img = cv2.imread("tests/game1/001.jpg")
##corners = np.array([[55, 119], [41, 521], [449, 619], [450, 18]],
##                   dtype = 'int32')
##board_size = 19
##
##mtx = np.identity(3)
##mtx[0,2] = img.shape[1]//2
##mtx[1,2] = img.shape[0]//2

img = cv2.imread('tests/calibrate/1.jpg')
corners = np.array([[132, 318], [177, 506], [387, 408], [307, 218]],
                   dtype=np.int32)
data = np.load('camera_params.npz')
mtx = data['mtx']
dist = data['dist']
board_size = 7

rvec, tvec, inliers, t = pose.get_pose(corners, board_size, mtx, dist, True)
img = pose.draw_pose(img, board_size, corners, t, rvec, tvec, mtx, dist)

cv2.imshow('pose', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
