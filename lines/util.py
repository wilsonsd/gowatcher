import numpy as np
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
