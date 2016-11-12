import numpy as np
import cv2
import itertools

def square(a):
    return itertools.product(range(a), range(a))

def intersect_lines(line1, line2):
    dx1 = line1[0,0] - line1[1,0]
    dy1 = line1[0,1] - line1[1,1]
    dx2 = line2[0,0] - line2[1,0]
    dy2 = line2[0,1] - line2[1,1]
    #print(line1)
    #print(line2)
    x = (dx1*dx2*line1[0,1] - dx2*dy1*line1[0,0] - dx1*dx2*line2[0,1]
         + dx1*dy2*line2[0,0]) / (dx1*dy2 - dx2*dy1)
    y = (dy1*dy2*line1[0,0] - dy2*dx1*line1[0,1] - dy1*dy2*line2[0,0]
         + dy1*dx2*line2[0,1]) / (dy1*dx2 - dy2*dx1)
    return np.array([x, y], dtype='float32')

def get_board_mask(shape, corners):
    board_mask = np.zeros(shape[0:2], dtype=np.uint8)
    #print(corners)
    board_mask = cv2.fillConvexPoly(board_mask, corners[:,::-1], 1)
    return board_mask

def same_color(gw_color, gm_color):
    """True is go_mill color is same as gowatcher color"""
    
    return ((gm_color == 'w' and gw_color == 1) or \
            (gm_color == 'b' and gw_color == 2) or \
            (gm_color is None and gw_color == 0))

def color2str(col):
    return 'white' if col == 'w' else 'black'

def other_color(gm_color):
    return 'w' if gm_color == 'b' else 'b'

def add_stone(lst, stone):
    return np.int32(np.append(lst, np.array([stone]),
                              axis = 0))

def remove_stone(lst, stone):
    return lst[(lst != np.array(stone)).any(1)]

def get_board_corners(grid):
    bsmo = grid.shape[0] - 1
    return np.array([grid[0,0], grid[0,bsmo], grid[bsmo,bsmo], grid[bsmo,0]],
                    dtype = grid.dtype)
