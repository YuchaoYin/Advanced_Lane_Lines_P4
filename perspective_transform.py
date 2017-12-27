import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from camera_calibration import calibration, undistort
from image_binarization import get_binary_image


def perspective(img, verbose=False):

    h, w = img.shape[:2]

    src = np.float32([[1075, 684],
                      [200, 684],
                      [570, 460],
                      [710, 460]])
    dst = np.float32([[w*4/5, h],
                      [w/5, h],
                      [w/5, 0],
                      [w*4/5, 0]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)

    if verbose == True:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img, cmap='gray')
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(warped, cmap='gray')
        ax2.set_title('Warped Result', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

    return warped, M, Minv

if __name__ == '__main__':
    cali_path = '../CarND-Advanced-Lane-Lines/camera_cal'
    pickle_file = '../CarND-Advanced-Lane-Lines/pick'
    img = mpimg.imread('../CarND-Advanced-Lane-Lines/test_images/straight_lines1.jpg')
    #plt.imshow(img)
    #plt.show()
    ret, mtx, dist, rvecs, tvecs = calibration(img, pickle_file, cali_path, verbose=False)
    dst = undistort(img, mtx, dist, verbose=False)
    binary_image = get_binary_image(dst, 3, True)
    warped, M = perspective(binary_image, verbose=True)