import numpy as np
import cv2
import find_grid

#img = cv2.imread('example.jpg')
img = cv2.imread('image.jpg')
#img = cv2.imread('tests/t3/t3_000.jpg')
#if img.shape[0] > 1000 or img.shape[1] > 1000:
#    m = img.shape[0] if img.shape[0] > 1000 else img.shape[1]
#    img.
if img.shape[0] > 700 or img.shape[1] > 1200:
    img = cv2.resize(img, (int(img.shape[0]*0.5), int(img.shape[1]*0.5)))
find_grid.find_grid(img, 19, None, True)
