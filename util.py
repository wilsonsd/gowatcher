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

def same_color(gm_color, gw_color):
    """True if go_mill color is same as gowatcher color"""
    
    return ((gm_color == 'w' and gw_color == 1) or \
            (gm_color == 'b' and gw_color == 2) or \
            (gm_color is None and gw_color == 0))

def color2str(col):
    if col == 'w' or col == 1:
        return 'white'
    elif col == 'b' or col == 2:
        return 'black'
    else:
        return 'unknown-color'

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

def draw_stones(img, white, black, size):
    draw_stones_one_color(img, white, size, (255, 255, 255))
    draw_stones_one_color(img, black, size, (0, 0, 0))

def draw_stones_one_color(img, stones, size, color):
    color2 = (255 - color[0], 255 - color[1], 255 - color[2])
    for i in range(stones.shape[0]):
        cv2.circle(img, tuple(stones[i,::-1]), size//6+2, color2, -1)
        cv2.circle(img, tuple(stones[i,::-1]), size//6, color, -1)

def make_roi_rectangle(im, center, radius):
    size = 2*radius + 1
    height, width = im.shape[0], im.shape[1]
    top = center[0] - radius
    bottom = center[0] + radius
    left = center[1] - radius
    right = center[1] + radius
    top2 = max(top, 0)
    bottom2 = min(bottom, height-1)
    left2 = max(left, 0)
    right2 = min(right, width-1)

    roi_center = (radius - (top2 - top),
                  radius - (left2 - left))

    return im[top:bottom, left:right,...], roi_center

    
