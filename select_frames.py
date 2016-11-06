import numpy as np
import cv2
from time import sleep

class FrameSelector:

    def __init__(self, *a, **k):
        if len(a) > 0:
            if isinstance(a[0], int) or isinstance(a[0], str):
                self.cap = cv2.VideoCapture(*a, **k)
            else:
                # assume it's a cv2.VideoCapture object
                self.cap = a[0]                

        self.roi = None
        self.mask = None
        self.last_good_frame = None
        self.last_good_frame_bw = None
        self.edges = None
        self.window = 20
        self.board_average = 0
        self.edge_average = 0
        self.decay_rate = 0.2
        self.recent_board_maximum = 0
        self.difference_threshold = 0.5

    def __getattr__(self, n):
        return getattr(self.cap, n)

    def set_roi(self, roi, mask):
        '''Set ROI and mask for the actual board area.'''

        self.roi = roi
        self.mask = mask
        if self.mask is not None:
            kernel = np.ones((3,3), dtype=np.int32)
            kernel[1,1] = -8
            self.edges = cv2.filter2D(255*self.mask, cv2.CV_16S, kernel)
            ret, self.edges = cv2.threshold(self.edges, 0, 1,
                                            cv2.THRESH_BINARY)
            self.edges = np.uint8(self.edges)

            #print('mask max', mask.max(), 'edges max', self.edges.max())
            #cv2.imshow('mask', 255*mask)
            #cv2.imshow('edges', 255*self.edges)
            

    def difference(self, im):
        #print(im.shape, self.last_good_frame.shape)
        diff = cv2.absdiff(self.last_good_frame, im)
        channels = cv2.split(diff)
        diff = cv2.max(cv2.max(channels[0], channels[1]), channels[2])
        ret, diff = cv2.threshold(diff, 30, 1, cv2.THRESH_BINARY)
        ker = np.ones((5,5), dtype=np.uint8)
        diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, ker)

##        if len(im.shape) == 3 and im.shape[2] > 1:
##            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
##        diff = cv2.absdiff(self.last_good_frame_bw, im)
##        ret, diff = cv2.threshold(diff, 35, 1, cv2.THRESH_BINARY)
##        ker = np.ones((5,5), dtype=np.uint8)
##        diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, ker)

        return diff

    def initialize(self):
        '''Watch the video for a while to initialize recordkeeping. Blocks.

        msec - number of milliseconds to watch the board to compute averages.
        delay - number of milliseconds to wait between observations.'''


        ret, example_frame = self.read()
        self.last_good_frame = example_frame
        self.last_good_frame_bw = cv2.cvtColor(
            example_frame, cv2.COLOR_BGR2GRAY)

        for i in range(self.window):
            ret, frame = self.read()
            self.accumulate_difference(frame)

    def accumulate_difference(self, im):
        diff = self.difference(im)
        self.diff = diff

        if self.edges is None:
            edges = np.ones(diff.shape, dtype='uint8')
            edges[2:-2,2:-2] = 0
        else:
            edges = self.edges
        edge_s = (edges * diff).sum()

        if self.mask is None:
            s = diff.sum()
        else:
            s = (self.mask*diff).sum()

        self.last_board_average = self.board_average
        self.board_average = self.board_average * self.decay_rate + s
        self.last_edge_average = self.edge_average
        self.edge_average = self.edge_average * self.decay_rate + edge_s

        return self.board_average, self.edge_average

    def check(self):
        '''Check if the next frame is a good candidate.  If so, return it.'''

        ret, im = self.read()
        if not ret:
            return False, None

        #im_bw = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        self.accumulate_difference(im)

        if self.board_average > self.recent_board_maximum:
            self.recent_board_maximum = self.board_average
            return False, im
        elif self.board_average < self.recent_board_maximum \
                                  * self.difference_threshold \
             and \
             self.board_average > self.last_board_average:
            self.last_good_frame = im
            self.last_good_frame_bw = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            self.board_average = 0
            self.last_board_average = 0
            self.edge_average = 0
            self.last_edge_average = 0
            self.recent_board_maximum = 0
            return True, im
        else:
            return False, im
        
##        if interesting_difference(self.last_good_frame,
##                                  self.last_good_frame_bw,
##                                  im, im_bw,
##                                  self.roi, self.mask, self.edges):
##            self.last_good_frame = im
##            self.last_good_frame_bw = im_bw
##            return True, im
##        else:
##            return False, im

def interesting_difference(im1, im1bw, im2, im2bw, roi, mask, edges):
        #take diff with last_good_frame
        #if diff is very small, ignore (no changes yet)
        #if difference region is connected to edge, ignore
        #   (someone is reaching over board)
        #otherwise, probably worth processing!

    #diff = im2bw - im1bw
    #diff = cv2.compare(np.float64(diff), 3*stdevs, cv2.CMP_GE)

    diff = cv2.absdiff(im2bw, im1bw)
    ret, diff = cv2.threshold(diff, 35, 1, cv2.THRESH_BINARY)
    ker = np.ones((5,5), dtype=np.uint8)
    diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, ker)
    
    cv2.imshow('diff', 255*diff)

    if edges is None:
        edges = np.ones(diff.shape, dtype='uint8')
        edges[2:-2,2:-2] = 0
    edge_s = (edges * diff).sum()

    #cv2.imshow('board diff', mask*diff)
    #cv2.imshow('edge diff', edges*diff)
    
    if mask is None:
        s = diff.sum()
    else:
        s = (mask*diff).sum()
    return s > 20 and edge_s < 5 #magic numbers!

if __name__ == '__main__':
    import test_select_frames
