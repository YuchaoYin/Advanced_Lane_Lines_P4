import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import pickle

def calibration(img, pickle_file, cali_path, verbose=False):

    # Check if calibration already computed
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as load_file:
            data = pickle.load(load_file)
        objpoints = data['objpoints']
        imgpoints = data['imgpoints']
    else:
        # Array to store object points and image points
        objpoints = []
        imgpoints = []

        for images in os.listdir(cali_path):

            # Read in a image
            image = mpimg.imread(os.path.join(cali_path, images))

            # Prepare object points like (0,0,0),(1,0,0)...
            objp = np.zeros((6*9,3),np.float32)
            objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            # If corners are found, add object points, image points
            if ret == True:
                imgpoints.append(corners)
                objpoints.append(objp)

                if verbose == True:
                    # Draw and display
                    cv2.drawChessboardCorners(image, (9, 6), corners, ret)
                    cv2.imshow('Draw Corners', image)


        with open(pickle_file, 'wb') as dump_file:
            data = {'objpoints':objpoints, 'imgpoints':imgpoints}
            pickle.dump(data, dump_file)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return ret, mtx, dist, rvecs, tvecs

def undistort(img, mtx, dist, verbose=False):

    dst = cv2.undistort(img, mtx, dist, None, mtx)

    if verbose == True:
        plt.imshow(dst)
        plt.title('Undistorted Image')
        plt.show()
        cv2.imwrite('/home/yiy6szh/CarND-Advanced-Lane-Lines/output_images/undistorted_test_image.jpg', dst)

    return dst

if __name__ == '__main__':
    cali_path = '/home/yiy6szh/CarND-Advanced-Lane-Lines/camera_cal'
    pickle_file = '/home/yiy6szh/CarND-Advanced-Lane-Lines/pick'
    img = mpimg.imread('/home/yiy6szh/CarND-Advanced-Lane-Lines/test_images/test2.jpg')
    plt.imshow(img)
    plt.show()
    ret, mtx, dist, rvecs, tvecs = calibration(img, pickle_file, cali_path, verbose=False)
    dst = undistort(img, mtx, dist, verbose=True)
    #cv2.imwrite('/home/yuchao/CarND-Advanced-Lane-Lines/output_images/original_image.jpg', img)
    #cv2.imwrite('/home/yiy6szh/CarND-Advanced-Lane-Lines/output_images/undistorted_test_image.jpg', dst)


