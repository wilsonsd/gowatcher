import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

img1 = cv2.imread('tests/game1/012.jpg',0)          # queryImage
img2 = cv2.imread('tests/game1/013.jpg',0) # trainImage

# Initiate SIFT detector
#sift = cv2.xfeatures2d.SIFT_create()
sift = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 0
FLANN_INDEX_LSH = 6
#index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

plt.imshow(img3, 'gray'),plt.show()


##
##img = cv2.imread('image.jpg',0)
##
### Initiate FAST object with default values
##fast = cv2.FastFeatureDetector_create()
##
### find and draw the keypoints
##kp = fast.detect(img,None)
##img2 = cv2.drawKeypoints(img, kp,img, color=(255,0,0))
##
### Print all default params
####print("Threshold: ", fast.getInt('threshold'))
####print( "nonmaxSuppression: ", fast.getBool('nonmaxSuppression'))
####print( "neighborhood: ", fast.getInt('type'))
####print( "Total Keypoints with nonmaxSuppression: ", len(kp))
##
##cv2.imshow('fast_true.png',img2)
##
##### Disable nonmaxSuppression
####fast.setBool('nonmaxSuppression',0)
####kp = fast.detect(img,None)
####
####print( "Total Keypoints without nonmaxSuppression: ", len(kp))
####
####img3 = cv2.drawKeypoints(img, kp, color=(255,0,0))
####
####cv2.imshow('fast_false.png',img3)
##cv2.waitKey(0)
