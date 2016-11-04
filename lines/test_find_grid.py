import cv2
import numpy as np
import find_grid

BOARDSIZE = 19


def to_tuple(a):
    return tuple(map(tuple, a))

def draw_stones(img, stones, color):
    for i in range(stones.shape[0]):
        #print(stones[i,:])
        cv2.circle(img, tuple(stones[i]), 2, color)

img = cv2.imread("image.jpg")
lines = find_grid.find_grid(img, BOARDSIZE)

cv2.polylines(img, np.int32(lines[0,:,:,:].reshape(-1,2,1,2)), False, (0,255,0))
cv2.polylines(img, np.int32(lines[1,:,:,:].reshape(-1,2,1,2)), False, (0, 0, 255))

#cv2.line(img, tuple(lines[0,5,0,:]), tuple(lines[0,5,1,:]), (255, 255, 255))
#black_stones, white_stones = find_stones(img, lines, BOARDSIZE)

#draw_stones(img, black_stones, (0,0,0))
#draw_stones(img, white_stones, (255, 255, 255))

grid = find_grid.get_grid_intersections(lines, BOARDSIZE)
        
grid = grid.reshape((-1, 2))
draw_stones(img, grid, (255, 255, 255))


cv2.imshow('grid', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
