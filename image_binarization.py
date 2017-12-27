import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import sys

'''
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(binary_output, cmap='gray')
ax2.set_title('Thresholded Gradient', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
'''
def abs_sobel_thresh(img, orient, sobel_kernel, thresh):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    else:
        print('wrong argument!')
        sys.exit(1)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return grad_binary

def mag_thresh(img, sobel_kernel, mag_thresh):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    return mag_binary

def dir_threshold(img, sobel_kernel, thresh):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary = np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    return dir_binary

def color_threshold(img, thresh):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = hls[:, :, 2]
    S_binary = np.zeros_like(S)
    S_binary[(S > thresh[0]) & (S_binary <= thresh[1])] = 1

    return S_binary

def yellow_threshold(img, thresh):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    b = lab[:,:,2]
    b_binary = np.zeros_like(b)
    b_binary[(b > thresh[0]) & (b <= thresh[1])] = 1

    return  b_binary

def white_threshold(img, thresh):
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    l = luv[:,:,0]
    l_binary = np.zeros_like(l)
    l_binary[(l > thresh[0]) & (l <= thresh[1])] = 1

    return l_binary

def morphology(img, ksize):
    kernel = np.ones(ksize, np.uint8)
    closing = cv2.morphologyEx(img.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    return closing

def get_binary_image(img, ksize, verbose=False):
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(0.7, 1.3))
    S_binary = color_threshold(img, thresh=(170, 255))
    b_binary = yellow_threshold(img, thresh=(155, 200))
    l_binary = white_threshold(img, thresh=(225, 255))
    # Combine thresholds
    combined = np.zeros_like(dir_binary)
    #combined = np.logical_or(gradx, S_binary)
    combined = np.logical_or(b_binary, l_binary)

    # Apply morphology
    combined = morphology(combined, ksize=(5,5))

    if verbose == True:
        plt.imshow(combined, cmap='gray')
        plt.title('B_binary_L_binary_combined')
        plt.show()

    return combined
#----------------------------------------------------------------------------


if __name__ == '__main__':

    img = mpimg.imread('../CarND-Advanced-Lane-Lines/test_images/test4.jpg')
    ksize = 3
    binary_image = get_binary_image(img, ksize, verbose=True)