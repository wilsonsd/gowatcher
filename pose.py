import numpy as np
import cv2

def get_pose(corners, board_size, cam_matrix = None, distortion = None,
             debug = False):
    if cam_matrix is None:
        cam_matrix = np.identity(3)

    bsmo = board_size - 1
    objp = np.array([[0,0,0], [0, bsmo, 0], [bsmo, bsmo, 0], [bsmo, 0, 0]],
                    dtype=np.float32)
    err_min = 99999999
    best_t = 1
    best_rvec, best_tvec, best_inliers = None, None, None
    for t in (1.077, 1/1.077, 1.038, 1/1.038, 1):
        modified_objp = objp * np.array([1,t,1])
#        rvec, tvec, inliers = cv2.solvePnPRansac(
#            modified_objp, corners, cam_matrix, distortion)
        ret, rvec, tvec = cv2.solvePnP(
            modified_objp, np.float32(corners), cam_matrix, distortion)
        inliers = np.arange(len(corners))

        reproj_error = reprojection_error(
            modified_objp, corners, rvec, tvec, cam_matrix, distortion)

        if debug:
            print('t', t, '  reproj error:', reproj_error)
        
        if reproj_error < err_min:
            err_min, best_t = reproj_error, t
            best_rvec, best_tvec, best_inliers = rvec, tvec, inliers

    if debug:
        print('chose t', best_t, '  with error', err_min)

    return best_rvec, best_tvec, best_inliers, best_t

def reprojection_error(objp, imgp, rvec, tvec, cam_mtx, dist):
    tot_error = 0
    imgpoints2, _ = cv2.projectPoints(
        objp, rvec, tvec, cam_mtx, dist)
##    print('imgp')
##    print(imgp)
##    print('imgpoints2')
##    print(imgpoints2[:,0,:])
##    print('difference')
##    print(imgp - imgpoints2[:,0,:])
##    print('norm')
##    print(np.linalg.norm(imgp - imgpoints2[:,0,:], axis=1))
    error = np.linalg.norm(imgp-imgpoints2[:,0,:], axis=1).sum()
    return error

def compute_offsets(grid, board_size, t, rvec, tvec, cam_mtx,
                    dist = None, stone_height=9/22):
    objp = np.zeros((board_size**2, 3), dtype=np.float32)
    objp[:,0:2] = np.mgrid[0:board_size,0:board_size].T[:,:,::-1].reshape(-1,2)
    objp = objp * np.array([1, t, 1], dtype=np.float32)
    objp[:,2] = -stone_height
    imgp, jac = cv2.projectPoints(objp, rvec, tvec, cam_mtx, dist)
    return imgp[:,0,:].reshape(board_size,board_size,2)

def draw_pose(img, board_size, corners, t, rvec, tvec, cam_mtx, dist = None):
    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]])
    imgp, jac = cv2.projectPoints(axis, rvec, tvec, cam_mtx, dist)
    cv2.line(img, tuple(corners[0].ravel())[::-1],
             tuple(imgp[0].ravel())[::-1], (255, 0, 0), 3)
    cv2.line(img, tuple(corners[0].ravel())[::-1],
             tuple(imgp[1].ravel())[::-1], (0, 255, 0), 3)
    cv2.line(img, tuple(corners[0].ravel())[::-1],
             tuple(imgp[2].ravel())[::-1], (0, 0, 255), 3)
    return img

if __name__ == '__main__':
    import test_pose
