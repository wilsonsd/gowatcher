import numpy as np
import cv2

im = cv2.imread('tests/game1/006.jpg')
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 5,
              param1=50, param2=30)
circles = np.uint16(np.around(circles))
for i in circles[0,:100]:
    # draw the outer circle
    cv2.circle(im,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(im,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('detected circles',im)
cv2.waitKey(0)
cv2.destroyAllWindows()
