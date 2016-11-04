import cv2
import numpy as np

def draw_lines(img, lines, thickness=1):
    cv2.polylines(img, np.int32(lines[0,:,:,::-1].reshape(-1,2,1,2)),
                  False, (0,255,0), thickness)
    cv2.polylines(img, np.int32(lines[1,:,:,::-1].reshape(-1,2,1,2)),
                  False, (0, 0, 255), thickness)

def draw_stones(img, stones, color, thickness=1):
    #print('stones to draw', stones.shape[0])
    #print(stones)
    #print(stones.shape)
    for i in range(stones.shape[0]):
        #print(stones[i,:])
        cv2.circle(img, tuple(stones[i,::-1]), 2, color, thickness)

def draw_hough_lines(img, lines, color, thickness=1):
    for rho,theta in lines[:]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 2000*(-b))
        y1 = int(y0 + 2000*(a))
        x2 = int(x0 - 2000*(-b))
        y2 = int(y0 - 2000*(a))

        cv2.line(img,(x1,y1),(x2,y2),color,thickness)
