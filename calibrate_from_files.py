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

images = glob.glob('tests\\calibrate\\*.jpg')
#cap = cv2.VideoCapture(0)
#for i in range(10):
#    cap.read()

for fname in images:
#count = 0
#while count < 10:
    img = cv2.imread(fname)
#    ret, img = cap.read()
#    k = cv2.waitKey(30)
#    if k == ord(' '):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points (after refining them)
    if ret:
        print('corners in image', fname)
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
        cv2.imshow('corners', img)
        cv2.waitKey(500)
    else:
        print('no corners in image', fname)
        cv2.imshow('bad image', img)
        cv2.waitKey(50)

#cv2.waitKey(0)
cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = \
     cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

images = glob.glob('tests\\calibrate\\*.jpg')

for fname in images:
    img = cv2.imread(fname)
    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    print('result', fname)
    cv2.imshow('result',dst)
    cv2.waitKey(50)

cv2.destroyAllWindows()

print('distortion parameters')
print(dist)

np.savez('camera_params', dist=dist, mtx=mtx, rvecs=rvecs, tvecs=tvecs)
