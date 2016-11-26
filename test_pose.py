import numpy as np
import cv2
import pose
import find_grid
import util
import find_stones

##img = cv2.imread("tests/game1/001.jpg")
##corners = np.array([[55, 119], [41, 521], [449, 619], [450, 18]],
##                   dtype = 'int32')
##board_size = 19
##
##mtx = np.identity(3)
##mtx[0,2] = img.shape[0]//2
##mtx[1,2] = img.shape[1]//2
##mtx[0,0] = max(img.shape[0],img.shape[1])
##mtx[1,1] = mtx[0,0]
##dist = None

##img = cv2.imread('tests/calibrate/1.jpg')
##corners = np.array([[132, 318], [177, 506], [387, 408], [307, 218]],
##                   dtype=np.int32)
##board_size = 7

img = cv2.imread('tests/photos/1.jpg')
corners = np.array([[75, 251], [164, 542], [445, 383], [275, 51]],
                   dtype=np.int32)
board_size = 9

data = np.load('camera_params.npz')
mtx = data['mtx']
#dist = data['dist']
dist = None

rvec, tvec, inliers, t = pose.get_pose(corners, board_size, mtx, dist, True)
img = pose.draw_pose(img, board_size, corners, t, rvec, tvec, mtx, dist)

lines = find_grid.find_grid(img, board_size, corners)
grid = find_grid.get_grid_intersections(lines, board_size)
offsets = pose.compute_offsets(grid, board_size, t, rvec, tvec, mtx, dist)


finder = find_stones.StoneFinder(board_size, lines, grid,
                                 np.zeros((0,2), dtype=np.int32),
                                 np.zeros((0,2), dtype=np.int32),
                                 offsets)
finder.draw_stone_masks(img)

for i,j in util.square(board_size):
    pt1, pt2 = tuple(grid[i,j,::-1].ravel()), tuple(offsets[i,j,::-1].ravel())
    cv2.line(img, pt1, pt2, (0, 255, 0), 2)


cv2.imshow('pose', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
