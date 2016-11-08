import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

#images = glob.glob('tests/calibrate/*.jpg')
cap = cv2.VideoCapture(0)
for i in range(10):
    cap.read()

#for fname in images:
count = 0
while count < 10:
#    img = cv2.imread(fname)
    ret, img = cap.read()
    k = cv2.waitKey(30)
    if k == ord(' '):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv2.imwrite('tests/calibrate/' + str(count) + '.jpg', img)
            img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
            cv2.imshow('corners', img)
            count += 1
    elif k == ord('q'):
        break
    cv2.imshow('img',img)

cap.release()
cv2.destroyAllWindows()
