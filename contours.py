import numpy as np
import cv2

im = cv2.imread('example.jpg')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

imgrayf = np.float32(imgray)
imgrayf = cv2.normalize(imgrayf, imgrayf, 0, 1, cv2.NORM_MINMAX)
M = np.ones((5,5), dtype='float32')
M[2,2]=-24
imedge = cv2.filter2D(imgrayf, cv2.CV_32F, M)
ret, imedge = cv2.threshold(imedge, 0, 0, cv2.THRESH_TOZERO)
imedge = cv2.normalize(imedge, imedge, 0, 1, cv2.NORM_MINMAX)
#imedge = np.uint8(imedge*255)
#ret, imedge = cv2.threshold(imedge, 50, 255, cv2.THRESH_BINARY)
#imcan = cv2.Canny(imedge, 50, 150)
#im2, contours, hierarchy = cv2.findContours(imedge,
#                    cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(im, contours, 0, (0, 255, 0), 3)
ret, canny_input = cv2.threshold(imedge, 0, 0, cv2.THRESH_TOZERO)
canny_input = np.uint8(canny_input*255)
imcan = cv2.Canny(imgray, 50, 150)
kernel = np.ones((3,3), np.uint8)
blackhat = cv2.morphologyEx(imcan, cv2.MORPH_BLACKHAT, kernel)

ret, canny_input = cv2.threshold(canny_input, 20, 255, cv2.THRESH_BINARY)

#cv2.imshow('contours', im)
#cv2.imshow('im2', im2)
cv2.imshow('edges', imedge)
cv2.imshow('canny', imcan)
cv2.imshow('blackhat', blackhat)
cv2.imshow('tresholded edges', canny_input)
cv2.waitKey(0)
cv2.destroyAllWindows()
