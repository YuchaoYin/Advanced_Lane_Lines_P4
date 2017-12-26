import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from perspective_transform import perspective
from camera_calibration import calibration, undistort
from image_binarization import get_binary_image
import collections

class Line():
    def __init__(self, buffer=10):
        self.detected = False
        self.current_fit_pixel = None
        self.current_fit_meter = None
        self.radius_of_curvature = None
        self.recent_fits_pixel = collections.deque(maxlen=buffer)
        self.recent_fits_meter = collections.deque(maxlen=2 * buffer)
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

    def parameter_update(self, new_fit_pixel, new_fit_meter, detected, buffer_clear):
        self.detected = detected
        if buffer_clear:
            self.recent_fits_meter = []
            self.recent_fits_pixel = []
        self.current_fit_meter = new_fit_meter
        self.current_fit_pixel = new_fit_pixel

        self.recent_fits_meter.append(self.current_fit_meter)
        self.recent_fits_pixel.append(self.current_fit_pixel)



def sliding_window_search(binary_warped, line_left, line_right, verbose=False):
    '''
    cali_path = '/home/yuchao/CarND-Advanced-Lane-Lines/camera_cal'
    pickle_file = '/home/yuchao/CarND-Advanced-Lane-Lines/pick'
    img = mpimg.imread('/home/yuchao/CarND-Advanced-Lane-Lines/test_images/test6.jpg')
    ret, mtx, dist, rvecs, tvecs = calibration(img, pickle_file, cali_path, verbose=False)
    dst = undistort(img, mtx, dist, verbose=False)
    binary_image = get_binary_image(dst, ksize=3, verbose=True)
    binary_warped, M = perspective(binary_image, verbose=False)
    '''
    h, w = binary_warped.shape[:2]
    ym_per_pix = 30. / 720.  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700.  # meters per pixel in x dimension

    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[np.int(h/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(h/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = h - (window+1)*window_height
        win_y_high = h - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    line_left.all_x = nonzerox[left_lane_inds]
    line_left.all_y = nonzeroy[left_lane_inds]
    line_right.all_x = nonzerox[right_lane_inds]
    line_right.all_y = nonzeroy[right_lane_inds]

    detected = True
    if not list(line_left.all_x) or not list(line_left.all_y):
        left_fit_pixel = line_left.current_fit_pixel
        left_fit_meter = line_left.current_fit_meter
        detected = False
    else:
        line_left_pixel = np.polyfit(line_left.all_y, line_left.all_x, 2)
        line_left_meter = np.polyfit(line_left.all_y*ym_per_pix, line_left.all_x*xm_per_pix, 2)

    if not list(line_right.all_x) or not list(line_right.all_y):
        right_fit_pixel = line_right.current_fit_pixel
        right_fit_meter = line_right.current_fit_meter
        detected = False
    else:
        line_right_pixel = np.polyfit(line_right.all_y, line_right.all_x, 2)
        line_right_meter = np.polyfit(line_right.all_y * ym_per_pix, line_right.all_x * xm_per_pix, 2)

    line_left.parameter_update(line_left_pixel, line_left_meter, detected=detected, buffer_clear=False)
    line_right.parameter_update(line_right_pixel, line_right_meter, detected=detected, buffer_clear=False)


    # Generate x and y values for plotting
    ploty = np.linspace(0, h - 1, h)
    left_fitx = line_left_pixel[0] * ploty ** 2 + line_left_pixel[1] * ploty + line_left_pixel[2]
    right_fitx = line_right_pixel[0] * ploty ** 2 + line_right_pixel[1] * ploty + line_right_pixel[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    if verbose == True:
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

    return line_right, line_left, out_img

def line_search_previous(binary_warped, line_left, line_right, verbose=False):

    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!

    h, w = binary_warped.shape[:2]
    ym_per_pix = 30. / 720.  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700.  # meters per pixel in x dimension



    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100

    left_fit = line_left.current_fit_pixel
    right_fit = line_right.current_fit_pixel

    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
    left_fit[1]*nonzeroy + left_fit[2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
    right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    line_left.all_x = nonzerox[left_lane_inds]
    line_left.all_y = nonzeroy[left_lane_inds]
    line_right.all_x = nonzerox[right_lane_inds]
    line_right.all_y = nonzeroy[right_lane_inds]

    detected = True
    if not list(line_left.all_x) or not list(line_left.all_y):
        left_fit_pixel = line_left.current_fit_pixel
        left_fit_meter = line_left.current_fit_meter
        detected = False
    else:
        line_left_pixel = np.polyfit(line_left.all_y, line_left.all_x, 2)
        line_left_meter = np.polyfit(line_left.all_y*ym_per_pix, line_left.all_x*xm_per_pix, 2)

    if not list(line_right.all_x) or not list(line_right.all_y):
        right_fit_pixel = line_right.current_fit_pixel
        right_fit_meter = line_right.current_fit_meter
        detected = False
    else:
        line_right_pixel = np.polyfit(line_right.all_y, line_right.all_x, 2)
        line_right_meter = np.polyfit(line_right.all_y * ym_per_pix, line_right.all_x * xm_per_pix, 2)


    # Fit a second order polynomial to each
    line_left.parameter_update(line_left_pixel, line_left_meter, detected=detected, buffer_clear=False)
    line_right.parameter_update(line_right_pixel, line_right_meter, detected=detected, buffer_clear=False)

    # Generate x and y values for plotting
    ploty = np.linspace(0, h-1, h)
    left_fitx = line_left_pixel[0] * ploty ** 2 + line_left_pixel[1] * ploty + line_left_pixel[2]
    right_fitx = line_right_pixel[0] * ploty ** 2 + line_right_pixel[1] * ploty + line_right_pixel[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)

    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin,
                                  ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin,
                                  ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    if verbose == True:
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

    return line_left, line_right, out_img

def project_on_to_original_image(undist, binary_warped, Minv, line_left, line_right, verbose=False):

    h, w = undist.shape[:2]
    left_fit = np.mean(line_left.recent_fits_pixel, axis=0)
    right_fit = np.mean(line_right.recent_fits_pixel, axis=0)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Generate x and y values for plotting
    ploty = np.linspace(0, h-1, h)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (w, h))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    if verbose == True:
        plt.imshow(result)
        plt.title('Original (undistorted) image with lane area drawn')
        plt.show()
    return result

if __name__ == '__main__':
    line_left = Line(buffer=10)
    line_right = Line(buffer=10)
    cali_path = '/home/yuchao/CarND-Advanced-Lane-Lines/camera_cal'
    pickle_file = '/home/yuchao/CarND-Advanced-Lane-Lines/pick'
    img = mpimg.imread('/home/yuchao/CarND-Advanced-Lane-Lines/test_images/test2.jpg')
    ret, mtx, dist, rvecs, tvecs = calibration(img, pickle_file, cali_path, verbose=False)
    dst = undistort(img, mtx, dist, verbose=False)
    binary_image = get_binary_image(dst, ksize=3, verbose=False)
    warped, M, Minv = perspective(binary_image, verbose=False)
    line_left, line_right, img_out = sliding_window_search(warped, line_left, line_right,  verbose=False)
    result = project_on_to_original_image(dst, warped, Minv, line_left, line_right, verbose=True)