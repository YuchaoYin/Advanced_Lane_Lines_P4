import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from camera_calibration import calibration, undistort
from moviepy.editor import VideoFileClip
from image_binarization import get_binary_image
from perspective_transform import perspective
from line_find import sliding_window_search, line_search_previous, project_on_to_original_image, Line
import numpy as np
import cv2

counter = 0 # counter of processed frames
line_left = Line(buffer=10)
line_right = Line(buffer=10)
def pipline(frame):
    global line_left, line_right, counter

    undistorted = undistort( frame, mtx, dist, verbose=False)
    binary_image = get_binary_image(undistorted, ksize=3, verbose=False)
    warped, M, Minv = perspective(binary_image, verbose=False)
    if counter > 0 and line_left.detected and line_right.detected:
        line_right, line_left, out_img = line_search_previous(warped, line_left, line_right
                                                               , verbose=False)
    else:
        line_left, line_right, out_img = sliding_window_search(warped, line_left, line_right
                                                              , verbose=False)

    draw = project_on_to_original_image(undistorted,  warped, Minv
                                        , line_left, line_right, verbose=False)
    # add curvature
    R_left, R_right, R_mean = curvature(line_left, line_right, draw)
    cv2.putText(draw, 'Radius of Curvature: {:.02f}(m)'.format(R_mean), (100, 60)
                , cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    # add position offset
    position_offset = vehicle_position(line_left, line_right, draw)
    cv2.putText(draw, 'Vehicle position offset from center: {:.02f}(m)'.format(position_offset), (100, 120)
                , cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    counter += 1
    return draw

def curvature(line_left, line_right, frame):
    y_eval = frame.shape[0]/2
    [A_left, B_left,_] = np.mean(line_left.recent_fits_meter, axis=0)
    R_left = ((1 + (2 * A_left * y_eval + B_left) ** 2) ** 1.5) / np.absolute(2 * A_left)
    [A_right, B_right,_] = np.mean(line_right.recent_fits_meter, axis=0)
    R_right = ((1 + (2 * A_right * y_eval + B_right) ** 2) ** 1.5) / np.absolute(2 * A_right)
    R_mean = np.mean([R_left, R_right])
    return R_left, R_right, R_mean

def vehicle_position(line_left, line_right, frame):
    w = frame.shape[1]
    # calculate the distance between two lines
    if line_right.detected and line_left.detected:
        left_bottom = np.mean(line_left.all_x[line_left.all_y > 0.95 * line_left.all_y.max()])
        right_bottom = np.mean(line_right.all_x[line_right.all_y > 0.95 * line_right.all_y.max()])
        width = right_bottom - left_bottom
        position_offset = left_bottom + width/2 - frame.shape[1]/2
        position_offset *= 3.7/700.

    return position_offset



if __name__ == '__main__':
    cali_path = '/home/yuchao/CarND-Advanced-Lane-Lines/camera_cal'
    pickle_file = '/home/yuchao/CarND-Advanced-Lane-Lines/pick'
    img_cal = mpimg.imread('/home/yuchao/CarND-Advanced-Lane-Lines/camera_cal/calibration2.jpg')
    ret, mtx, dist, rvecs, tvecs = calibration(img_cal, pickle_file, cali_path, verbose=False)
    #img = mpimg.imread('/home/yiy6szh/CarND-Advanced-Lane-Lines/test_images/test2.jpg')
    #dst = undistort(img, mtx, dist, verbose=True)

    video = '/home/yuchao/CarND-Advanced-Lane-Lines/project_video.mp4'
    clip = VideoFileClip(video)
    clip1 = clip.fl_image(pipline)
    out_video = '/home/yuchao/CarND-Advanced-Lane-Lines/out_video.mp4'
    clip1.write_videofile(out_video, audio=False)