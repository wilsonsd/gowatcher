import cv2
import numpy as np
import util
import debug_util
from math import pi, degrees, radians
import math
from operator import itemgetter

corners = []

HOUGH_DEGREE_BIN_SIZE = 0.5 #should evenly divide 180
HOUGH_BINS = int(180/HOUGH_DEGREE_BIN_SIZE)

def corner_click(event,x,y,flags,param):
    global corners
    if event == cv2.EVENT_LBUTTONDBLCLK:
        corners = np.append(corners, np.float32([[y, x]]), axis=0)
        print('Added corner number', corners.shape[0], (y, x))

def getLines(c, size):
    lines = np.zeros((2, size, 2, 2))

    for i in range(size):
        t = float(i)/(size-1)
        lines[0,i,0,:] = np.array((1-t)*c[0,:] + t*c[3,:])
        lines[0,i,1,:] = np.array((1-t)*c[1,:] + t*c[2,:])
        lines[1,i,0,:] = np.array((1-t)*c[0,:] + t*c[1,:])
        lines[1,i,1,:] = np.array((1-t)*c[3,:] + t*c[2,:])

    return lines

def find_grid(img, size, given_corners = None, find_lines_auto = False):
    global corners
    if isinstance(img, type(cv2.VideoCapture)):
        img = img.read()

    if find_lines_auto:
        corners = find_corners(img, size)
    else:
        if given_corners is None:
            corners = np.zeros((0, 2), dtype='float32')
            cv2.namedWindow('click window')
            cv2.setMouseCallback('click window', corner_click)
            
            cv2.imshow('click window', img)

            while(corners.shape[0] != 4):
                cv2.waitKey(20)

            cv2.destroyWindow('click window')
        else:
            corners = given_corners

    corners2 = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype='float32')
    M = cv2.getPerspectiveTransform(corners, corners2)
    points2 = get_points(size)

    points2 = np.array([points2])
    points = cv2.perspectiveTransform(points2, np.linalg.inv(M))

    return points.reshape((2, size, 2, 2))

def get_points(size):
    pts = np.zeros((size*4, 2), dtype='float32')
    ul = np.array([0, 0])
    ur = np.array([0, 1])
    ll = np.array([1, 0])
    lr = np.array([1, 1])

    dx = 1.0/(size-1)
    for i in range(size):
        t = i*dx
        pts[2*i,:] = (1-t)*ul + t*ll
        pts[2*i+1,:] = (1-t)*ur + t*lr
        pts[2*size+2*i,:] = (1-t)*ul + t*ur
        pts[2*size+2*i+1,:] = (1-t)*ll + t*lr
    #print(pts)
    #print(pts.reshape((2,size, 2, 2)))
    return pts

def get_grid_intersections(lines, size):
    grid = np.zeros((size, size, 2), dtype='float32')
    for i in range(size):
        for j in range(size):
            grid[i,j,:] = util.intersect_lines(lines[0,i,:,:], lines[1,j,:,:])
    return grid



def find_corners(img, board_size):
    # convert to gray, do initial filtering
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

##    imgrayf = np.float32(imgray)
##    imgrayf = cv2.normalize(imgrayf, imgrayf, 0, 1, cv2.NORM_MINMAX)
##    M = np.ones((5,5), dtype='float32')
##    M[2,2]=-24
##    imedge = cv2.filter2D(imgrayf, cv2.CV_32F, M)
##    ret, imedge = cv2.threshold(imedge, 0, 0, cv2.THRESH_TOZERO)
##    imedge = cv2.normalize(imedge, imedge, 0, 1, cv2.NORM_MINMAX)
##    imedge = np.uint8(255*imedge)
##    ret, imedge = cv2.threshold(imedge, 30, 255, cv2.THRESH_BINARY)
    imedge = cv2.Canny(imgray, 100, 200)

    kernel5 = np.ones((5,5), np.uint8)
    kernel3 = np.ones((3,3), np.uint8)
    imedge = cv2.morphologyEx(imedge, cv2.MORPH_CLOSE, kernel5)
    imedge = cv2.erode(imedge, kernel3)
    

    lines = cv2.HoughLines(imedge, 1, radians(HOUGH_DEGREE_BIN_SIZE),
                           int(max(imedge.shape)/7))[:,0,:]
    max_rho = cv2.reduce(np.absolute(lines[:,0]), 0, cv2.REDUCE_MAX)[0,0]
    hough_img, lines = make_hough_image(lines, max_rho)
    hough_lines = cv2.HoughLines(hough_img, 5, radians(5), 10)[:,0,:]

    #keep the first two hough lines that are "well separated",
    #since the vertical and horizontal board lines should have
    #very different angles.
    #prefer hough lines that don't slant too much
    #hough_lines = hough_lines[hough_lines[:,1].argsort()]
    good_lines = []
    first_line = -1
##    for i, line in enumerate(hough_lines):
##        if (line[0] > 5 and line[0] < HOUGH_BINS - 15) or \
##           (line[1] > np.radians(10) and line[1] < np.radians(345)):
##            first_line = i
##            good_lines.append(line)
##            break
    good_lines.append(hough_lines[0])
    first_line = 0
    if len(good_lines) == 0:
        raise ValueError("No good collection of lines found.")

    for line in hough_lines[first_line+1:]:
        line_dist = abs(x_intercept_rt(line) - x_intercept_rt(good_lines[0]))
#        line_dist = projective_dist_lines_rt(good_lines[0], line)
#        print(line_dist)
        if line[0] > 0 and \
           20/HOUGH_DEGREE_BIN_SIZE <  line_dist \
           and line_dist < 160/HOUGH_DEGREE_BIN_SIZE:
#        if line_dist > pi/2:
            good_lines.append(line)
            break
    print('selected hough lines', good_lines)
    board_lines = []
    # only keep lines that are near the hough lines.
    for i, line in enumerate(good_lines):
        board_lines.append(filter_hough_lines(lines, line, max_rho))
        board_lines[i] = board_lines[i][:100]
    
    #debug_util.draw_hough_lines(hough_img, good_lines, 255)
    #cv2.imshow('hough lines', np.transpose(hough_img))
    #cv2.waitKey(0)
    #cv2.destroyWindow('hough lines')
    
    # find lines with similar projective distance
##    dists.append(np.zeros((board_lines[0].shape[0],board_lines[0].shape[0])))
##    dists.append(np.zeros((board_lines[1].shape[0],board_lines[1].shape[0])))
##    for i in range(2):
##        for j in range(board_lines[i].shape[0]):
##            for k in range(board_lines[i].shape[0]):
##                #print('(',j,',',k,')')
##                dists[i][j,k] = projective_dist_lines_rt(
##                    board_lines[i][j], board_lines[i][k])

##    LINE_DIST_TOLERANCE = 0.05
##    dists = []
##    medians = []
##    selected_lines = []
##    for i in range(2):
##        print('board_line nans', np.count_nonzero(np.isnan(board_lines[i])))
##        print('board_line shape', board_lines[i].shape)
##        dists.append(np.ma.array(projective_dist_lines_rt(board_lines[i])))
##        dists[i].mask = (1 == np.eye(dists[i].shape[0]))
##        dists[i].mask = np.logical_or(dists[i].mask,
##                                       dists[i] < 10)
##        medians.append(np.ma.median(dists[i]))
##        print('median', medians[i])
##        print('min', np.ma.minimum(dists[i]), 'max', np.ma.maximum(dists[i]))
##        #print('distances', dists[i].shape)
##        #print(dists[i])
##        selected_lines.append(
##            board_lines[i][ np.ma.where(
##                np.ma.logical_and(medians[i]*(1-LINE_DIST_TOLERANCE) < dists[i],
##                                  medians[i]*(1+LINE_DIST_TOLERANCE) > dists[i])
##                )[0] ] )

    selected_lines = []
    for i in range(len(board_lines)):
        selected_lines.append(lines_with_same_vanishing_point(board_lines[i]))
    ########################################
    ### TODO: sort differences into bins (bin width a percentage of
    ### distance from min dist to max dist), choose bin with
    ### most lines.  (Also discard bins that correspond to
    ### a very small number of pixels.)


    for i in range(len(board_lines)):
        debug_util.draw_hough_lines(img, board_lines[i],
                                    (0, 255*i, 255*(1-i)), 1)
        debug_util.draw_hough_lines(img, selected_lines[i], (255, 255, 255), 1)

    #hough_img_copy = hough_img.copy()
    debug_util.draw_hough_lines(hough_img, good_lines, 255)
    cv2.line(hough_img, (0, int(max_rho)), (2*HOUGH_BINS, int(max_rho)), 150, 2)
    for i in range(len(board_lines)):
        for r,t in board_lines[i]:
            cv2.circle(hough_img, (t_to_bin(t),int(r+max_rho)), 2, 150, 1)
    cv2.imshow('edges', cv2.resize(imedge, None, fx=1, fy=1))
    cv2.imshow('hough with lines', np.transpose(hough_img))
    cv2.imshow('lines', cv2.resize(img, None, fx=1, fy=1))
    #cv2.imshow('hough', np.transpose(hough_img_copy))
    cv2.waitKey(0)
    cv2.destroyWindow('lines')
    #cv2.destroyWindow('hough')
    cv2.destroyWindow('hough with lines')
    cv2.destroyWindow('edges')

    # pick two hough lines that are well-separated (the only way this would
    # not be the case is if we're viewing from a very low angle)

def t_to_bin(t):
    return (np.degrees(t)/HOUGH_DEGREE_BIN_SIZE).astype('int32')

def make_hough_image(lines, max_rho):
    hough_image = np.zeros((int(2*max_rho+1), 2*HOUGH_BINS), dtype='uint8')
    new_lines = np.zeros((lines.shape[0] * 2,2))
    for i, l in enumerate(lines):
        if pi/9 < l[1] < 17*pi/9:
            hough_image[int(l[0]+max_rho),t_to_bin(l[1])] = 255
        if pi/9 < l[1]+pi < 17*pi/9:
            hough_image[int(-l[0]+max_rho), t_to_bin(l[1])+HOUGH_BINS] = 255
        new_lines[2*i] = l
        new_lines[2*i+1] = np.array([-l[0], l[1]+pi])

    return hough_image, new_lines

def filter_hough_lines(lines, line, max_rho):
    c, s = math.cos(line[1]), math.sin(line[1])
    x0, y0 = c*line[0], s*line[0]

    return lines[ 10 > np.absolute( np.add(c*(t_to_bin(lines[:,1]) - x0),
                                           s*(lines[:,0]+max_rho-y0)))]


def calc_vanishing_points(lines):
    n = np.transpose(
        np.array([np.cos(lines[...,1]), np.sin(lines[...,1]), -lines[...,0]]))
    nnorm = np.linalg.norm(n, None, 1)
    nunit = n/nnorm[...,np.newaxis]

    vps = np.cross(nunit[:,np.newaxis,:], nunit)
    idx = np.triu_indices(n.shape[0])
    return vps[idx]

def cluster_vanishing_points(vps, n):
    #return n clusters and list of outliers
    #for each VP:
    #   find all VP within 1 deg of vp.
    COS_ONE_DEGREE = 0.999847695156391239
    closeness = np.absolute(np.inner(vps[:,np.newaxis,:],vps)) \
                        > COS_ONE_DEGREE
    counts = np.sum(closeness, axis=1)
    max_idx = np.argmax(counts)
    cluster = vps[closeness[max_idx],:]
    
    return cluster

def lines_with_same_vanishing_point(lines):
    #calculate vanishing points
    n = np.transpose(
        np.array([np.cos(lines[...,1]), np.sin(lines[...,1]), -lines[...,0]]))
    
    vps = np.cross(n[:,np.newaxis,:], n)
    vps = vps/np.linalg.norm(vps, None, 2, True)
    idx = np.triu_indices(n.shape[0])
    flat_vps = vps[idx]

    #find best vanishing point
    COS_ONE_DEGREE = 0.999847695156391239
    COS_HALF_DEGREE = 0.99999999
    closeness = np.absolute(np.inner(flat_vps[:,np.newaxis,:],flat_vps)) \
                        > COS_HALF_DEGREE
    counts = np.sum(closeness, axis=2)[:,0]
    max_idx = np.argmax(counts)
    best_vp = flat_vps[max_idx]

    #find lines that share this vanishing point.
    good_lines = lines[
        np.where(np.absolute(np.inner(vps, best_vp)) > COS_HALF_DEGREE)[0]]
    return good_lines

    
    
##def projective_dist_lines_rt2(line1, line2):
##    #lines are
##    #0 = x cos t1 + y sin t1 - r1
##    #0 = x cos t2 + y sin t2 - r2
##    #so the lines in homogeneous coords are
##    # (cos t1)x + (sin t1) y + (-r1) z = 0
##    # (cos t2)x + (sin t2) y + (-r2) z = 0
##    u1, u2, u3 = math.cos(line1[...,1]), math.sin(line1[1]), -line1[0]
##    v1, v2, v3 = math.cos(line2[1]), math.sin(line2[1]), -line2[0]
##    
##    return math.acos(abs(
##        (u1*v1+u2*v2+u3*v3)
##        /
##        math.sqrt((u1*u1+u2*u2+u3*u3)*(v1*v1+v2*v2+v3*v3))))

def projective_dist_lines_rt3(lines):
    #vectorized
    #n1 = np.array([np.cos(line1[1]), np.sin(line1[1]), line1[0]])
    #n2 = np.array([np.cos(line2[1]), np.sin(line2[1]), line2[0]])
##    norms = np.linalg.norm(n1, axis=1)*np.linalg.norm(n2, axis=1)
##    dot = np.inner(n1, n2)
##
##    return np.sqrt(np.square(norms) - np.square(dot))/(norms + dot)
    dims = len(lines.shape)
    n1 = np.transpose(
        np.array([np.cos(lines[...,1]), np.sin(lines[...,1]), -lines[...,0]]))

    norms = np.linalg.norm(n1, axis=dims-1)[...,np.newaxis] \
            * np.linalg.norm(n1, axis=dims-1)
    dot = np.inner(n1, n1)
    dist = np.ma.sqrt(np.square(norms) - np.square(dot)) \
        / \
        (norms + dot)
    return dist

def projective_dist_lines_rt(lines):
    dims = len(lines.shape)
    n1 = np.transpose(
        np.array([np.cos(lines[...,1]), np.sin(lines[...,1]), -lines[...,0]]))
    I = np.cross(n1[:,np.newaxis,:], n1)
    #nonsense values on the diagonal, so we don't divide by zero.
    I[np.eye(I.shape[0]).astype('bool')] = np.array([1,0,0])
    Inorm = np.sqrt(np.sum(I*I, 2))
    Iunit = I/Inorm[:,:,np.newaxis]

    k = np.array([0,0,1])
    
    pvect = k - I[:,:,2,np.newaxis]*Iunit/Inorm[:,:,np.newaxis]
    pvect = np.array([pvect, np.cross(Iunit, pvect)])
    #p1 = np.cross(lines[np.newaxis, np.newaxis,:,:], pvect)
    #p2 = np.cross(lines[np.newaxis,:,np.newaxis,:], pvect)
    p1 = np.where(I[np.newaxis,:,:,2,np.newaxis] == 0,
                  np.cross(lines[np.newaxis,np.newaxis,:,:], I[np.newaxis]),
                  np.cross(lines[np.newaxis,np.newaxis,:,:], pvect))
    p2 = np.where((I[np.newaxis,:,:,2,np.newaxis] == 0),
                  np.cross(lines[np.newaxis:,np.newaxis,:], I[np.newaxis]),
                  np.cross(lines[np.newaxis,:,np.newaxis,:], pvect))
    pe1 = p1[:,:,:,0:2]/p1[:,:,:,2,np.newaxis]
    pe2 = p2[:,:,:,0:2]/p2[:,:,:,2,np.newaxis]
    dists = np.where(p1[:,:,:,2] == 0,
                     np.abs(lines[np.newaxis,:,np.newaxis,0]
                            - lines[np.newaxis:,0]),
                     np.sqrt(np.sum(np.square(pe1-pe2), axis=3)))
    print('dists')
    print(dists[0])
    print(dists[1])
                     
    return np.minimum(dists[0], dists[1])


def x_intercept_rt(line):
    return line[0]/math.cos(line[1])

class OrthogonalVanishingPointModel:

    def __init__(self, data):
        vp1 = np.cross(data[0], data[1])
        vp1 = vp1/np.linalg.norm(vp1, keepdims=True)
        vp2 = np.cross(vp1, data[2])
        vp2 = vp2/np.linalg.norm(vp2, keepdims=True)
        return np.vstack(vp1,vp2)

    def get_error(self, data, model):
        
            
if __name__ == '__main__':
    import test_auto_find_grid
    
