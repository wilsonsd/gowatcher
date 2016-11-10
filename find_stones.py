"""find_stone.py - a module containing classes and helper functions
for finding stones in an image."""

import numpy as np
import cv2
import util
import math
from itertools import product


class StoneFinder:
    """A class for finding stones in an image"""
    
    def __init__(self, board_size, lines, \
                 grid, white, black, offsets = None):
        """Initialize StoneFinder object.

        board_size -- the size of the board (e.g. 9, 13, 19)
        lines -- numpy array with shape (2,size,2,2) containing
                 the start and end points of the horizontal and
                 vertical board gridlines ("horizontal" and
                 "vertical" is arbitrary, since we may be
                 viewing the board at an angle).  Produced by
                 find_grid.py
        grid -- numpy array with shape (size, size, 2) containing
                the image coordinates of grid intersections.  Produced
                by find_grid.py
        white -- a list of (row,col) pairs of known white stones
        black -- a list of (row,col) pairs of known black stones
        offsets -- numpy array with shape (size, size, 2) giving, for each
                   board intersection, the offset pixel to use,
                   based on the perspective at which we are viewing
                   the board.
        """

        self.last_gray_image = None
        self.board_size = board_size
        self.lines = lines
        self.grid = np.int32(grid)
        self.white = white
        self.black = black
        self.found_in_last_frame = False
        self.offsets = offsets if offsets is not None else grid

        self.ycc_avgs = np.zeros((self.board_size, self.board_size, 3))
        self.hsv_avgs = np.zeros((self.board_size, self.board_size, 3))
        self.disuniformity = np.zeros((self.board_size, self.board_size))
        self.diff_avgs = np.zeros((self.board_size, self.board_size))

        self.was_obscured = None                                  

        #Approximate stone size
        #TODO: calculate vertical and horizontal sizes independently.
        self.stone_size = np.zeros((board_size, board_size), np.uint8)
        bsmo = board_size-1
        def dist(i,j,a,b):
            return int(np.linalg.norm(grid[i,j]-grid[i+a,j+b]))
        self.stone_size[0,0] = max(dist(0,0,1,0), dist(0,0,0,1), dist(0,0,1,1))
        self.stone_size[0,bsmo] = max(dist(0,bsmo,1,0), dist(0,bsmo,0,-1),
                                      dist(0,bsmo, 1, -1))
        self.stone_size[bsmo,0] = max(dist(bsmo,0,-1,0), dist(bsmo,0,0,1),
                                      dist(bsmo,0, -1, 1))
        self.stone_size[bsmo,bsmo] = max(dist(bsmo,bsmo,-1,0),
                        dist(bsmo,bsmo,0,-1), dist(bsmo,bsmo,-1,-1))
        for i in range(1,board_size-1):
            self.stone_size[0,i] = max(dist(0,i,1,-1), dist(0,i,1,0),
                                       dist(0,i,1,1))
            self.stone_size[bsmo,i] = max(dist(bsmo,i,-1,-1),
                            dist(bsmo,i,-1,0), dist(bsmo,i,-1,1))
            self.stone_size[i,0] = max(dist(i,0,-1,1), dist(i,0,0,1),
                                       dist(i,0,1,1))
            self.stone_size[i,bsmo] = max(dist(i,bsmo,-1,-1),
                            dist(i,bsmo,0,-1), dist(i,bsmo,1,-1))
        for i,j in product(range(1,bsmo), range(1, bsmo)):
            self.stone_size[i,j] = max(dist(i,j,-1,-1), dist(i,j,-1,0),
                                       dist(i,j,-1,1), dist(i,j,0,-1),
                                       dist(i,j,0,1), dist(i,j,1,-1),
                                       dist(i,j,1,0), dist(i,j,1,1))
        self.middle_size = self.stone_size[board_size//2,board_size//2]

        #prepare the morphological operation kernel.
        kernel_radius = self.middle_size//5
        self.kernel = np.zeros((2*kernel_radius+1, 2*kernel_radius+1),
                               np.uint8)
        for i,j in util.square(2*kernel_radius+1):
            if (i-kernel_radius)*(i-kernel_radius) + \
               (j - kernel_radius) * (j-kernel_radius) < \
               kernel_radius * kernel_radius:
                self.kernel[i,j] = 1
        

        #prepare the roi for averaging rgb & hs values near intersection
        self.roi_middle = int(np.amax(self.stone_size)/4)
        self.roi_size = self.roi_middle*2 + 1
        self.stone_roi = np.zeros((self.board_size,self.board_size,
                                   self.roi_size, self.roi_size),
                                  dtype='uint8')
        for i, j in util.square(self.roi_size):
            dist_sq_grid = (i-self.roi_middle)*(i-self.roi_middle)+\
               (j-self.roi_middle)*(j-self.roi_middle)
            delta = self.roi_middle + (self.offsets - grid) \
                    - np.array([i,j], dtype=np.uint8)
            dist_sq_offset = (delta*delta).sum(axis=2)
            rad_sq = (self.stone_size/4) * (self.stone_size/4)
            self.stone_roi[np.logical_and(0.05*rad_sq < dist_sq_grid,
                                          dist_sq_offset < 0.4*rad_sq),
                           i,j] = 255
#           if 0.1*rad_sq < dist_sq and \
#               dist_sq < rad_sq:
#                self.stone_roi[i,j] = 255

        #prepare spiral for computing "disuniformity"
        self.spiral = []
        SPIRAL_POINTS = 3*self.middle_size
        theta_step = 2*math.pi/SPIRAL_POINTS
        radial_step = float(self.middle_size/(SPIRAL_POINTS*8)) # 8 is a magic number
        theta, r = 0, 0
        self.spiral.append((0,0))
        for i in range(SPIRAL_POINTS):
            pt = (int(r * math.sin(theta)),
                                int(r * math.cos(theta)))
            if not pt == self.spiral[-1]:
                self.spiral.append((int(r * math.sin(theta)),
                                    int(r * math.cos(theta))))
            theta += theta_step
            r += radial_step
        
##        print('spiral points:', len(self.spiral))
##        pic = np.zeros((self.roi_size, self.roi_size), dtype='uint8')
##        for y, x in self.spiral:
##            cv2.circle(pic, (x+ self.roi_middle, y+self.roi_middle), 0, (255))
##        cv2.imshow('spiral', pic)
##        print(self.spiral)

    def set_stones(self, white, black):
        self.white = white
        self.black = black

    def set_last_gray_image(self, im):
        self.last_gray_image = im
        self.current_gray = im

    def set_image(self, im):
        self.img_bgr = im
        self.img_ycc = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)
        self.img_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

        if self.found_in_last_frame:
            self.last_gray_image = self.current_gray
        self.found_in_last_frame = False

        #calculate difference from last image.
        self.current_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        if self.last_gray_image is None:
            self.diff_im = 255*np.ones(im.shape[0:2], dtype='uint8')
        else:
            self.diff_im = cv2.absdiff(self.current_gray, self.last_gray_image)
            ret, self.diff_im = cv2.threshold(
                self.diff_im,25, 255, cv2.THRESH_BINARY)
            self.diff_im = cv2.morphologyEx(self.diff_im,
                                            cv2.MORPH_CLOSE,
                                            self.kernel)
            self.diff_im = cv2.morphologyEx(self.diff_im,
                                            cv2.MORPH_OPEN,
                                            self.kernel)
            #self.diff_im = cv2.dilate(self.diff_im, kernel)

            k = 2
            board_corners = np.array([self.grid[0,0] - k*(self.grid[1,1]
                                                          - self.grid[0,0]),
                                      self.grid[0,-1] - k*(self.grid[1,-2]
                                                           - self.grid[0,-1]),
                                      self.grid[-1,-1] - k*(self.grid[-2,-2]
                                                            - self.grid[-1,-1]),
                                      self.grid[-1,0] - k*(self.grid[-2,1]
                                                           - self.grid[-1,0])],
                                     dtype=np.int32)            
            brd = np.zeros(self.diff_im.shape, dtype=np.uint8)
            brd = cv2.fillConvexPoly(brd, board_corners[:,::-1], 1)
            #brd = cv2.dilate(brd, self.kernel, iterations=8)
            #cv2.imshow('brd', brd*255)
            #cv2.imshow('raw diff', self.diff_im)
            output = cv2.connectedComponentsWithStats(self.diff_im)
            num = output[0]
            components = output[1]
            stats = output[2]

            if self.was_obscured is None:
                self.was_obscured = np.zeros(self.diff_im.shape, np.uint8)
            obscured = np.zeros(self.was_obscured.shape, np.uint8)
            for i in range(1,num):
                #if the component touches the edge, or if it's bigger
                #than several stones
                if np.any(np.logical_and(components == i, 1-brd)) or \
                   stats[i,cv2.CC_STAT_AREA] > \
                   3*self.middle_size * self.middle_size:
                    self.diff_im[components == i] = 0
                    obscured[components == i] = 1
            #cv2.imshow('obscured pre', 255*obscured)
            self.diff_im[ np.logical_and(np.logical_not(obscured),
                                         self.was_obscured) ] = 255
            #obscured = np.logical_and(np.logical_not(self.was_obscured),
            #                          obscured)
            self.was_obscured = obscured

            #cv2.imshow('obscured post', np.uint8(obscured))            
            cv2.imshow('diff', self.diff_im)


        grid = self.grid

        for i, j in util.square(self.board_size):
            try:
                #print('self.roi_middle', self.roi_middle)
                #print('self.stone_roi[', i, ',', j, ']', self.stone_roi[i,j].shape)
                self.ycc_avgs[i,j,:] = np.array(cv2.mean( 
                    self.img_ycc[grid[i,j,0]-self.roi_middle:
                                 grid[i,j,0]+self.roi_middle,
                                 grid[i,j,1]-self.roi_middle:
                                 grid[i,j,1]+self.roi_middle],
                    self.stone_roi[i,j]))[0:3]
                self.hsv_avgs[i,j,:] = np.array(cv2.mean(
                    self.img_hsv[grid[i,j,0]-self.roi_middle:
                                 grid[i,j,0]+self.roi_middle,
                                 grid[i,j,1]-self.roi_middle:
                                 grid[i,j,1]+self.roi_middle],
                    self.stone_roi[i,j]))[0:3]
                self.diff_avgs[i,j] = cv2.mean(
                    self.diff_im[grid[i,j,0]-self.roi_middle:
                                 grid[i,j,0]+self.roi_middle,
                                 grid[i,j,1]-self.roi_middle:
                                 grid[i,j,1]+self.roi_middle],
                    self.stone_roi[i,j])[0]
            except:
                print('index probably out of bounds', i, j,
                      self.roi_middle, grid[i,j])
                #cv2.circle(self.img_bgr, tuple(grid[i,j]), 5, (0,255,0),-1)
                raise

            last = self.img_ycc[self.spiral[0]]
            total = 0
            for spiral_offset in self.spiral:
                try:
                    cur = self.img_ycc[spiral_offset[0] + grid[i,j,0],
                                       spiral_offset[1] + grid[i,j,1]]
                    total += np.linalg.norm(cur - last)
                    last = cur
                except:
                    print('spiral', i, j, spiral_offset,
                          (spiral_offset[0] + grid[i,j,0],
                           spiral_offset[1] + grid[i,j,1]))
                    print('image shape', self.img_ycc.shape)
                    #cv2.circle(self.img_bgr, tuple(grid[i,j]), 5, (0,255,0), -1)
                    raise

            self.disuniformity[i,j] = total/len(self.spiral)
        

    def calculate_features(self):
##        print(self.stone_roi)
##        print(img_ycc[self.grid[10,10,0]-self.roi_middle:
##            self.grid[10,10,0]+self.roi_middle,
##            self.grid[10,10,1]-self.roi_middle:
##            self.grid[10,10,1]+self.roi_middle][0,0])
##        print(np.array(cv2.mean(img_ycc[self.grid[10,10,0]-self.roi_middle:
##            self.grid[10,10,0]+self.roi_middle,
##            self.grid[10,10,1]-self.roi_middle:
##            self.grid[10,10,1]+self.roi_middle]))[0:3])
##        print(np.array(cv2.mean(img_ycc[self.grid[10,10,0]-self.roi_middle:
##            self.grid[10,10,0]+self.roi_middle,
##            self.grid[10,10,1]-self.roi_middle:
##            self.grid[10,10,1]+self.roi_middle]))[0:3].shape)

        #build array masks to ignore known stones
        self.stone_at = np.full((self.board_size,self.board_size),
                                 False, dtype=np.bool)
        self.stone_at[self.white[:,0], self.white[:,1]] = True
        self.stone_at[self.black[:,0], self.black[:,1]] = True
        
        big_mask = np.full((self.board_size,self.board_size, 3),
                           False, dtype=np.bool)
        big_mask[...] = self.stone_at[:,:,np.newaxis]
        #print(big_mask.shape)
        empty_ycc_masked = np.ma.array(self.ycc_avgs, mask = big_mask)
        empty_hsv_masked = np.ma.array(self.hsv_avgs, mask = big_mask)
        #self.empty_ycc_avg = np.mean(empty_ycc_masked, (0,1))
        self.empty_ycc_avg = empty_ycc_masked.mean((0,1))
        self.empty_hsv_avg = empty_hsv_masked.mean((0, 1))
        empty_disunif_masked = np.ma.array(self.disuniformity,
                                           mask = self.stone_at)
        self.empty_disunif_avg = empty_disunif_masked.mean((0, 1))
        #print(empty_disunif_masked.mask)

        # calculate mean Y, Cr, Cb, and Disuniformity for white, black, and
        # empty
        if len(self.white) != 0:
            self.white_ycc_avg = np.mean(
                self.ycc_avgs[self.white[:,0],self.white[:,1]], 0)
            self.white_hsv_avg = np.mean(
                self.hsv_avgs[self.white[:,0],self.white[:,1]], 0)
            self.white_disunif_avg = np.mean(
                self.disuniformity[self.white[:,0],self.white[:,1]], 0)
        else:
            self.white_ycc_avg = self.empty_ycc_avg * \
                                 np.array([1.1, 0.9, 1.1])
            self.white_hsv_avg = self.empty_hsv_avg
            self.white_disunif_avg = self.empty_disunif_avg
        if len(self.black) != 0:
            self.black_ycc_avg = np.mean(
                self.ycc_avgs[self.black[:,0],self.black[:,1]], 0)
            self.black_hsv_avg = np.mean(
                self.hsv_avgs[self.black[:,0],self.black[:,1]], 0)
            self.black_disunif_avg = np.mean(
                self.disuniformity[self.black[:,0],self.black[:,1]], 0)
        else:
            self.black_ycc_avg = self.empty_ycc_avg * \
                                 np.array([0.6, 1, 1])
            self.black_hsv_avg = self.empty_hsv_avg
            self.black_disunif_avg = self.empty_disunif_avg

        empty_ycc_diff = np.array(
            [self.ycc_avgs - self.empty_ycc_avg,
             self.ycc_avgs - self.white_ycc_avg,
             self.ycc_avgs - self.black_ycc_avg])
        empty_hsv_diff = np.array([
            self.hsv_avgs - self.empty_hsv_avg,
            self.hsv_avgs - self.white_hsv_avg,
            self.hsv_avgs - self.black_hsv_avg])
        empty_disunif_diff = np.array([
            self.disuniformity - self.empty_disunif_avg,
            self.disuniformity - self.white_disunif_avg,
            self.disuniformity - self.black_disunif_avg])
        
        self.features = \
            np.sqrt((empty_ycc_diff*empty_ycc_diff).sum(3) +
                    empty_hsv_diff[:,:,:,0]*empty_hsv_diff[:,:,:,0] +
                    empty_disunif_diff*empty_disunif_diff)

    def find_next_stone(self):
        """Return a ((row, col), color) tuple of the next stone, where color
           is 0 for white and 1 for black.  Returns None if there is no
           new stone. """

        #find grid points to check.
        points = np.where(self.diff_avgs > 20)

        #empties = np.array([points[0], points[1], 0,
        #                    self.features[0,points[0], points[1]]]).transpose()
        whites = np.zeros((len(points[0]), 4))
        whites[:,0] = points[0]
        whites[:,1] = points[1]
        whites[:,2] = 1
        whites[:,3] = self.features[1,points[0],points[1]]
        blacks = whites.copy()
        blacks[:,2] = 2
        blacks[:,3] = self.features[2,points[0],points[1]]
        empties = whites[:,0:3].copy()
        empties[:,2] = self.features[0,points[0],points[1]]
        
        #points_to_check = np.append(empties, whites, blacks)
        points_to_check = np.append(whites, blacks, axis=0)
        #print('whites', whites.shape, '\n', whites)
        #print('point_to_check shape', points_to_check.shape)
        points_to_check = points_to_check[points_to_check[:,3].argsort()]
        #print('points to check\n', points_to_check)

        for i, position in enumerate(points_to_check):
            row, col, color = np.int32(position[0:3])
            if self.stone_at[row,col]:
                continue
            value = position[3]
            #print('inspecting', position)
            like_empty = self.features[0,row,col] <= value
            close_like_empty = (6/5)*self.features[0,row,col] <= value
            stone_mean = np.mean(
                points_to_check[i+1:min(i+15,len(points_to_check)-1),3])
            stone_mean = 99999 if np.isnan(stone_mean) else stone_mean
            like_stone = value <=  stone_mean
            # condition 6c in photokifu paper, page 10, not implimented
            #print(self.features[0,row,col], stone_mean,
            #      'like_empty', like_empty, 'like_stone', like_stone)

            if not close_like_empty and not self.stone_at[row,col]:
                circle_goodness = self.like_circle(row, col)
                #print('circle goodness', circle_goodness)

                if (not like_empty and like_stone and circle_goodness <= 4) \
                   or \
                   (not close_like_empty and circle_goodness <= 2):
                    self.found_in_last_frame = True
                    return (row, col), color

        #if we didn't find any stones, look for removed stones
        empties = empties[empties[:,2].argsort()]
        for i, position in enumerate(empties):
            row, col, value = np.int32(position)
            #print('checking empty', row, col, value,
            #      self.features[(1,2),row,col].min())
            #print('self at', row, col, self.stone_at[row,col])
            if value < (2/3)* self.features[(1,2),row,col].min() \
               and self.stone_at[row,col]:
                self.found_in_last_frame = True
                return (row, col), 0
        
        #print(np.int32(self.diff_avgs))
        return None, None

    def like_circle(self, row, col):
        '''Return true if a circle is found near gridpoint (row, col).'''

        radius = int(self.stone_size[row,col]//2)
        region = self.current_gray[max(self.offsets[row,col,0] - radius,0) :
                                   min(self.offsets[row,col,0] + radius,
                                       self.diff_im.shape[0]-1),
                                   max(self.offsets[row,col,1] - radius,0) :
                                   min(self.offsets[row,col,1] + radius,
                                       self.diff_im.shape[1]-1)].copy()
        center = np.array([radius, radius])

        region = cv2.GaussianBlur(region, (5, 5), 0)
        circles = cv2.HoughCircles(region, cv2.HOUGH_GRADIENT, 1,
                                   self.stone_size[row,col]//2, param1=25,
                                   param2=5,
                                   minRadius=self.stone_size[row,col]//3,
                                   maxRadius = 4*self.stone_size[row,col]//3)
        #print('center', center, 'circles', circles, sep='\n')
        
        if circles is not None:
            icircles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in icircles:
                cv2.circle(region, (x, y), r, 255, 1)

        
        cv2.imshow('circle', region)
        if circles is None:
            return 999999
        else:
            circles = circles[0]
            #print('circles')
            #print(circles)
            #print(np.linalg.norm(center - circles[:,0:2], axis=1))
            #if the best circle is near the center, we're good.
            good_circles = np.where(self.stone_size[row,col]/3 >
                             np.linalg.norm(
                                 center - circles[:,0:2], axis=1))[0]
            #print('stone_size/2', self.stone_size[row,col]/2,
            #      'good circles at', good_circles)
            if good_circles.shape[0] > 0:
                return good_circles[0]
            else:
                return 99999

    def draw_stone_masks(self, im):
        for i,j in util.square(self.board_size):
            #print('roi_middle', self.roi_middle)
            #print('stone roi', self.stone_roi[i,j].shape)
            roi = im[self.grid[i,j,0]-self.roi_middle:
                     self.grid[i,j,0]+self.roi_middle+1,
                     self.grid[i,j,1]-self.roi_middle:
                     self.grid[i,j,1]+self.roi_middle+1]
            #print('roi', roi.shape)
            roi[self.stone_roi[i,j] > 0, :] = 255

        return im

if __name__ == "__main__":
    import test_find_stone_sequence
