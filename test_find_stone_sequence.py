import cv2
import numpy as np
import find_grid
import find_stones
import matplotlib.pyplot as plt
from debug_util import draw_stones


BOARDSIZE = 19
RUN_UNTIL = 1
pic = 0

corners = np.array([[55, 119], [41, 521], [449, 619], [450, 18]],
                   dtype = 'float32')
#white = np.array([(2, 3), (5, 2), (2, 14), (15, 13), (14, 16)], dtype='int32')
#black = np.array([(3, 16), (4, 15), (15, 2), (15, 4), (16, 15)], dtype='int32')
white = np.zeros((0,2), dtype='int32')
black = np.zeros((0,2), dtype='int32')
stone_size = 25

file = open('tests/game1.txt', 'r')
prefix = 'tests/'
#file = open('c:/sam/tests/game1.txt', 'r')
#prefix='c:/sam/tests/'
img = cv2.imread(prefix+file.readline().strip(), cv2.IMREAD_COLOR)

lines = find_grid.find_grid(img, BOARDSIZE,corners)
grid = find_grid.get_grid_intersections(lines, BOARDSIZE)
grid = np.int32(grid)

#print('shape of image', img.shape)
#print('grid')
#print(grid)

finder = find_stones.StoneFinder(BOARDSIZE, lines, grid,
               white, black)
finder.set_last_gray_image(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

##draw_stones(img, grid.reshape((-1,2)), (0, 255, 255), 2)
##cv2.imshow('game', img)
##cv2.waitKey(500)

cont = True

for fname in file:
    print('reading', fname)
    img = cv2.imread(prefix+fname.strip(), cv2.IMREAD_COLOR)
    img_copy = img.copy()
    pic += 1
    
    finder.set_image(img)
    finder.calculate_features()
    stone, color = finder.find_next_stone()
    
    found = 0
    while stone is not None:
        if color == 1:
            draw_stones(img_copy, np.array([grid[stone]]),
                        (0, 0, 0),
                        1+ 2*stone_size // 3 )
            draw_stones(img_copy, np.array([grid[stone]]),
                        (255, 255, 255),
                        2*stone_size // 3)
            white = np.int32(np.append(white, np.array([stone]), axis = 0))
            found += 1
        elif color == 2:
            draw_stones(img_copy, np.array([grid[stone]]),
                        (255, 255, 255),
                        1+2*stone_size // 3)
            draw_stones(img_copy, np.array([grid[stone]]),
                        (0, 0, 0),
                        2*stone_size//3)
            black = np.int32(np.append(black, np.array([stone]), axis = 0))
            found += 1
        elif color == 0:
            #need to remove a stone
            draw_stones(img_copy, np.array([grid[stone]]),
                        (0,255,0),
                        stone_size // 2)
            #print('white\n', white)
            #print('stone\n', np.array(stone))
            #print('noteq\n', (white != np.array(stone)).any(1))
            #print('new white\n', white[(white != np.array(stone)).any(1)])
            white = white[(white != np.array(stone)).any(1)]
            #if len(white) == 0:
            #    white = np.array((0,2), dtype=np.int32)
            black = black[(black != np.array(stone)).any(1)]
            #if len(black) == 0:
            #    black = np.array((0,2), dtype=np.int32)


        print('found stone at', stone, 'color', color)
        #print('white\n', white)
        #print(grid[np.transpose(white)[0],
        #           np.transpose(white)[1]])
        #print(grid[np.transpose(white)].shape)
        #print(white)
        if white.shape[0] > 0:
            draw_stones(img_copy, grid[white[:,0],
                                       white[:,1]],
                        (0, 0, 0), stone_size // 2)
            draw_stones(img_copy, grid[white[:,0],
                                       white[:,1]],
                        (255, 255, 255), stone_size // 3)
        if black.shape[0] > 0:
            draw_stones(img_copy, grid[black[:,0],
                                       black[:,1]],
                        (255, 255, 255), stone_size // 2)
            draw_stones(img_copy, grid[black[:,0],
                                       black[:,1]],
                        (0, 0, 0), stone_size // 3)

        cv2.imshow('game', img_copy)
        
        if pic >= RUN_UNTIL and cv2.waitKey(0) == ord('q'):
            cont = False
            break

        img_copy = img.copy()
        finder.set_stones(white, black)
        #print('white', white, 'black', black, sep='\n')
        finder.calculate_features()
        stone, color = finder.find_next_stone()

    if found == 0:
        if white.shape[0] > 0:
            draw_stones(img_copy, grid[white[:,0],
                                       white[:,1]],
                        (0, 0, 0), stone_size // 2)
            draw_stones(img_copy, grid[white[:,0],
                                       white[:,1]],
                        (255, 255, 255), stone_size // 3)
        if black.shape[0] > 0:
            draw_stones(img_copy, grid[black[:,0],
                                       black[:,1]],
                        (255, 255, 255), stone_size // 2)
            draw_stones(img_copy, grid[black[:,0],
                                       black[:,1]],
                        (0, 0, 0), stone_size // 3)
        cv2.imshow('game', img_copy)
        if pic >= RUN_UNTIL and cv2.waitKey(0) == ord('q'):
            cont = False
            break

    finder.set_last_gray_image(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    if not cont:
        break
    
    
file.close()
cv2.destroyAllWindows()
