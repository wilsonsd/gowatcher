import numpy as np
import cv2
import find_grid
import find_stones
import select_frames
import pose
import util
from gomill import gtp_states


class GoWatcher:

    def __init__(self, source, board_size, camera_calibration_file,
                 window_name = 'Game', rotate_pic = True, debug = False):
        self.source = source
        self.board_size = board_size
        self.calibration_file = camera_calibration_file
        self.cam_mtx = None
        self.distortion = None
        self.cap = None
        self.lines = None
        self.grid = None
        self.offsets = None
        self.corners = None
        self.finder = None
        self.window_name = window_name
        self.rotate_pic = rotate_pic
        self.debug = debug

        data = np.load(camera_calibration_file)
        self.cam_mtx = data['mtx']
        #self.distortion = data['dist']
        self.distortion = None

        self.white = np.zeros((0,2), dtype=np.int32)
        self.black = np.zeros((0,2), dtype=np.int32)

    def __del__(self):
        if self.cap is not None:
            self.cap.release()

    def initialize(self):        
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            cap.open()

        # read a few frames to make sure the camera is self-calibrated
        for i in range(10):
            cap.read()

        # let user position camera
        print("Position the camera, then hit space.")
        while cv2.waitKey(20) != ord(' '):
            ret, img = cap.read()
            cv2.imshow(self.window_name, img)

        # find the board grid
        ret, img = cap.read()
        self.lines = find_grid.find_grid(img, self.board_size)
        self.grid = find_grid.get_grid_intersections(
            self.lines, self.board_size)
        self.corners = util.get_board_corners(self.grid)
        rvec, tvec, inliners, t = pose.get_pose(self.corners,
                    self.board_size, self.cam_mtx, self.distortion)
        self.offsets = pose.compute_offsets(self.grid, self.board_size,
                    t, rvec, tvec, self.cam_mtx, self.distortion)

        self.corners = np.int32(self.corners)
        self.offsets = np.int32(self.offsets)
        board_mask = util.get_board_mask(img.shape[:2], self.corners)
        self.cap = select_frames.FrameSelector(cap)
        self.cap.set_roi(None, board_mask)
        self.cap.initialize()
        
        self.finder = find_stones.StoneFinder(self.board_size,
                    self.lines, self.grid, self.black, self.white,
                    self.offsets)
        self.finder.set_last_gray_image(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

        if self.debug:
            self.finder.draw_stone_masks(img)

            for i,j in util.square(self.board_size):
                pt1, pt2 = tuple(self.grid[i,j,::-1].ravel()), \
                           tuple(self.offsets[i,j,::-1].ravel())
                cv2.line(img, pt1, pt2, (0, 255, 0), 2)
            cv2.imshow('pose', img)
            cv2.waitKey(0)
            cv2.destroyWindow('pose')

        return True

    def play_move(self, game_state, color):
        ret, last_move = gtp_states.get_last_move(
            game_state.move_history, color)
        if not ret:
            return
        
        row, col = last_move
        last_color = util.other_color(color)
        print("Please place a %s stone on (%d, %d)" %
              (util.color2str(last_color), row+1, col+1))
        while True:
            move, new_color = self.wait_for_move(last_move)
            if row == move[0] and col == move[1] and \
               util.same_color(last_color, new_color):
                break
            print("Wrong placement.  Please place a %s stone on (%d, %d)."
                  % (util.color2str(last_color), row+1, col+1))
            print("You placed a %s stone on (%d, %d)"
                  % (util.color2str(new_color), move[0]+1, move[1]+1))

            if self.debug:
                print(last_move, move, last_color, new_color)
                print(row == move[0], col == move[1],
                      util.same_color(last_color, new_color))
                print(last_color == 'w', new_color == 1)

        self.add_stone(move, new_color)
        
    def get_move(self, color):
        print("Make a move, %s" % (util.color2str(color),))
        wrong_color = True

        while True:
            move, new_color = self.wait_for_move()
            if util.same_color(color, new_color):
                break
            print("Wrong color.  Please place a stone, %s." %
                  (util.color2str(color),))
            
        self.add_stone(move, new_color)
        result = gtp_states.Move_generator_result()
        result.move = move
        
        return result

    def add_stone(self, move, color):
        if color == 1:
            self.white = util.add_stone(self.white, move)
        elif color == 2:
            self.black = util.add_stone(self.black, move)
        self.finder.set_stones(self.white, self.black)
        

    def genmove(self, game_state, color):
        old_move = gtp_states.get_last_move(game_state.move_history, color)

        if self.debug:
            print("last move")
            print(gtp_states.get_last_move(game_state.move_history, color))

        self.play_move(game_state, color)
        ret = self.get_move(color)

        if self.debug:
            print("new move")
            print(ret)
            print(ret.move)

        return ret

    def wait_for_move(self, move_to_mark = None):
        found_one = False
        while not found_one:
            ret, img = self.cap.check()
            move = []
            
            if ret:
                self.finder.set_image(img.copy())
                white = self.white
                black = self.black
                while True:
                    self.finder.calculate_features()
                    stone, color = self.finder.find_next_stone()
                    move.append((stone, color))

                    if color == 1:
                        white = util.add_stone(white, stone)
                        found_one = True
                    elif color == 2:
                        black = util.add_stone(black, stone)
                        found_one = True
                    elif color == 0:
                        white = util.remove_stone(white, stone)
                        black = util.remove_stone(black, stone)
                        found_one = True
                    self.finder.set_stones(white, black)

                    if color is None:
                        break

            if move_to_mark is not None:
                cv2.circle(img,
                           tuple(self.grid[move_to_mark[0],move_to_mark[1]][::-1]),
                           7, (0,0,0), -1)
                cv2.circle(img,
                           tuple(self.grid[move_to_mark[0],move_to_mark[1]][::-1]),
                           5, (0,0,255), -1)
            self.draw_stones(img)
            if self.rotate_pic:
                cv2.imshow(self.window_name, img[::-1,::-1])
            else:
                cv2.imshow(self.window_name, img)
            if cv2.waitKey(20) == ord('q'):
                break

        return move[0]

    def draw_stones(self, img):
        util.draw_stones(img,
                         self.grid[self.white[:,0], self.white[:,1]],
                         self.grid[self.black[:,0], self.black[:,1]],
                         self.finder.middle_size)
        return img
    
