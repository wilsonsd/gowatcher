import numpy as np
import cv2
import select_frames
import find_grid
import find_stones
import util
from debug_util import *

BOARDSIZE = 9

#source = 'vid1.mp4'
#stone_size = 10
#grid_corners = np.array([[10, 150], [10, 462], [348, 468], [350, 146]],
#                   dtype=np.int32)
#board_corners = np.array([[2, 140], [2, 472], [358, 478], [358, 136]],
#                         dtype=np.int32)

#source = 'tests/go game.mp4'
source=0
stone_size=10
#grid_corners = np.array([[20, 95], [20, 391], [339, 393], [341, 93]],
#                        dtype=np.int32)
#board_corners = np.array([[7, 79], [7, 403], [354, 409], [354, 80]],
#                         dtype=np.int32)

cap = cv2.VideoCapture(source)

if not cap.isOpened():
    cap.open()

#throw away several frames so the camera can self-calibrate
for i in range(10):
    cap.read()

ret, frame = cap.read()

white = np.zeros((0,2), dtype='int32')
black = np.zeros((0,2), dtype='int32')

data = np.load('camera_params.npz')
mtx = data['mtx']
dist = data['dist']

lines = find_grid.find_grid(frame, BOARDSIZE)
grid = find_grid.get_grid_intersections(lines, BOARDSIZE)
grid = np.int32(grid)
board_corners = np.array([grid[0,0], grid[0,BOARDSIZE-1],
                          grid[BOARDSIZE-1,BOARDSIZE-1], grid[BOARDSIZE-1,0]],
                         dtype=np.int32)

rvec, tvec, inliers, t = pose.get_pose(board_corners, BOARDSIZE, mtx, dist)
offsets = post.compute_offsets(grid, BOARDSIZE, t, rvec, tvec, mtx, dist)

cap = select_frames.FrameSelector(cap)
cap.initialize()
cap.set_debug(True)

board_mask = np.zeros(frame.shape[0:2], dtype=np.uint8)
board_mask = cv2.fillConvexPoly(board_mask, board_corners[:,::-1], 1)
cap.set_roi(None, board_mask)

finder = find_stones.StoneFinder(BOARDSIZE, lines, grid,
                                white, black, offsets)
finder.set_last_gray_image(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
print("watching now")
found_one = False
try:
    while cap.isOpened():
        found_one = False
        ret, frame = cap.check()
        
        if ret:
            finder.set_image(frame.copy())
            while True:
                finder.calculate_features()
                stone, color = finder.find_next_stone()
                print('found stone', stone, color, 'at time',
                      cap.get(cv2.CAP_PROP_POS_MSEC))

                if color == 1:
                    white = np.int32(np.append(white, np.array([stone]),
                                               axis = 0))
                    found_one = True
                elif color == 2:
                    black = np.int32(np.append(black, np.array([stone]),
                                               axis = 0))
                    found_one = True
                elif color == 0:
                    white = white[(white != np.array(stone)).any(1)]
                    black = black[(black != np.array(stone)).any(1)]
                    found_one = True
                finder.set_stones(white, black)

                if color is None:
                    break

        if white.shape[0] > 0:
            draw_stones(frame, grid[white[:,0],
                        white[:,1]],
                        (0, 0, 0), stone_size // 2)
            draw_stones(frame, grid[white[:,0],
                        white[:,1]],
                        (255, 255, 255), stone_size // 3)
        if black.shape[0] > 0:
            draw_stones(frame, grid[black[:,0],
                        black[:,1]],
                        (255, 255, 255), stone_size // 2)
            draw_stones(frame, grid[black[:,0],
                        black[:,1]],
                        (0, 0, 0), stone_size // 3)

        cv2.imshow('game', frame)
        if cv2.waitKey(1) == ord('q'):
            break

        #if found_one:
        #    if cv2.waitKey(0) == ord('q'):
        #        break
        
        
except:
    raise

cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
