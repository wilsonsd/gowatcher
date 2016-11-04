import cv2
import numpy as np
import find_grid
import find_stones
import matplotlib.pyplot as plt
from debug_util import *
import random

BOARDSIZE = 19
TIMES_TO_TEST = 1

img = cv2.imread("tests/game1/001.jpg")
corners = np.array([[55, 119], [41, 521], [449, 619], [450, 18]],
                   dtype = 'float32')
black = np.array([[15, 15]], dtype=np.int32)
white = np.zeros((0,2), dtype=np.int32)

##img = cv2.imread("image.jpg")
##corners = np.array([[133, 404], [121, 1523], [1356, 1608], [1385, 384]],
##                   dtype = 'float32')
##white = np.array([(2, 3), (5, 2), (2, 14), (15, 13), (14, 16)], dtype='int32')
##black = np.array([(3, 16), (4, 15), (15, 2), (15, 4), (16, 15)], dtype='int32')

lines = find_grid.find_grid(img, BOARDSIZE,corners)
grid = find_grid.get_grid_intersections(lines, BOARDSIZE)
grid = np.int32(grid)

right = 0

def mark_stones():
    ymin, ymax = plt.ylim()
    for i,j in white:
        plt.plot((i*BOARDSIZE+j,i*BOARDSIZE+j), (ymin,ymax), color='red')
        
    for i,j in black:
        plt.plot((i*BOARDSIZE+j,i*BOARDSIZE+j), (ymin,ymax), color='black')


for i in range(TIMES_TO_TEST):
    removed = 0
    removed_color = random.randint(1,2)
    removed_color = 1 if removed_color == 2 and len(black) == 0 \
                    else removed_color
    removed_color = 2 if removed_color == 1 and len(white) == 0 \
                    else removed_color
    
    if removed_color == 1:
        removed = random.randint(0,white.shape[0]-1)
        removed_loc = white[removed]
    else:
        removed = random.randint(0, black.shape[0]-1)
        removed_loc = black[removed]

    #debug_img = img.copy()
    #debug_img = cv2.resize(debug_img, (500, 500))
    #cv2.imshow("original", debug_img)


    ##debug_img = img.copy()
    ##draw_lines(debug_img, lines)
    ##draw_stones(debug_img, grid.reshape((-1, 2)), (255, 255, 255), 2)
    ##debug_img = cv2.resize(debug_img, (500, 500))
    ##cv2.imshow('debug image', debug_img)

    finder = 0
    if removed_color == 1:
        finder = find_stones.StoneFinder(BOARDSIZE, lines, grid,
                   np.append(white[0:removed], white[removed+1:], axis=0),
                   black)
    else:
        finder = find_stones.StoneFinder(BOARDSIZE, lines, grid,
                   white,
                   np.append(black[0:removed], black[removed+1:], axis=0))

    finder.set_image(img)
    finder.calculate_features()
    stone, color = finder.find_next_stone()
    print('\nstone:', stone, ' color:', color)
    print('removed:', removed_loc, ' color:', removed_color)
    if removed_color == 1:
        if removed_color == color and \
           tuple(stone) == tuple(removed_loc):
            right += 1
    else:
        if removed_color == color and \
           tuple(stone) == tuple(removed_loc):
            right += 1

     
    if color is not None and color != 0:
        draw_stones(img, np.array([grid[stone]]), (0,255,0) if color == 1 else (0,255,0), 30)

    #draw_stones(img, grid[white[:,0], white[:,1]], (0,0,255), 30)
    #draw_stones(img, grid[black[:,0], black[:,1]], (0,255,0), 30)
    img = cv2.resize(img, (500, 500))
    cv2.imshow('new stone', img)
    cv2.waitKey(1)

    print(finder.features[1])


    ##img = cv2.resize(img, (600, 600))
    ##cv2.imshow('errors', img)


    #print('white ycc average', finder.white_ycc_avg)
    #print('black ycc average', finder.black_ycc_avg)
    #print('empty ycc average', finder.empty_ycc_avg)
    n = np.arange(361)

    xvals = np.arange(BOARDSIZE*BOARDSIZE)
    plt.subplot(811)
    plt.plot(xvals, finder.features[0].reshape((-1)))
    plt.ylabel('empty')
    plt.xlim(0, 361)
    mark_stones()

    plt.subplot(812)
    plt.plot(xvals, finder.features[1].reshape((-1)))
    plt.ylabel('white')
    plt.xlim(0, 361)
    mark_stones()

    plt.subplot(813)
    plt.plot(xvals, finder.features[2].reshape((-1)))
    plt.ylabel('black')
    plt.xlim(0, 361)
    mark_stones()

    plt.subplot(814)
    plt.plot(xvals, finder.ycc_avgs[:,:,0].reshape((-1)))
    plt.ylabel('Y')
    plt.xlim(0, 361)
    mark_stones()

    plt.subplot(815)
    plt.plot(xvals, finder.ycc_avgs[:,:,1].reshape((-1)))
    plt.ylabel('Cr')
    plt.xlim(0, 361)
    mark_stones()

    plt.subplot(816)
    plt.plot(xvals, finder.ycc_avgs[:,:,2].reshape((-1)))
    plt.ylabel('Cb')
    plt.xlim(0, 361)
    mark_stones()

    plt.subplot(817)
    plt.plot(xvals, finder.hsv_avgs[:,:,0].reshape((-1)))
    plt.ylabel('H')
    plt.xlim(0, 361)
    mark_stones()

    plt.subplot(818)
    plt.plot(xvals, finder.disuniformity.reshape((-1)))
    plt.ylabel('D')
    plt.xlim(0, 361)
    mark_stones()

    plt.show()

    cv2.destroyAllWindows()

    
##    draw_stones(img, np.array([grid[stone]]), (0,255,0) if color == 1 else (0,255,0), 30)
##    #draw_stones(img, grid[white[:,0], white[:,1]], (0,0,255), 30)
##    #draw_stones(img, grid[black[:,0], black[:,1]], (0,255,0), 30)
##    img = cv2.resize(img, (500, 500))
##    cv2.imshow('new stone', img)
##    cv2.waitKey(0)
##    cv2.destroyAllWindows()

print(right, '/', TIMES_TO_TEST, 'correct')
print(100*right/TIMES_TO_TEST, '% correct', sep='')
