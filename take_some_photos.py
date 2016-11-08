import numpy as np
import cv2
import glob

num_photos = 5
file_dir = 'tests/photos/'

#images = glob.glob('tests/calibrate/*.jpg')
cap = cv2.VideoCapture(0)
for i in range(10):
    cap.read()

#for fname in images:
count = 0
while count < num_photos:
#    img = cv2.imread(fname)
    ret, img = cap.read()
    k = cv2.waitKey(30)
    if k == ord(' '):
        cv2.imwrite(file_dir + str(count) + '.jpg', img)
        cv2.imshow('photo', img)
        count += 1
    elif k == ord('q'):
        break
    cv2.imshow('video',img)

cap.release()
cv2.destroyAllWindows()
