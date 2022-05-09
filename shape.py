

from __future__ import print_function
import cv2 as cv
import numpy as np


def thresh_callback(val):
    threshold = val
    # Detect edges using Canny
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    # Find contours
    contours, hierarchy = cv.findContours(canny_output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # Draw contours
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        cv.drawContours(drawing, contours, i, (0, 255, 0), 2,cv.LINE_8, hierarchy, 0)
    # Show in a window
    cv.imshow('Contours', drawing)
# Load source image
src = cv.imread("predict1.png")

# Convert image to gray and blur it
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
original = src_gray.copy()
cv.GaussianBlur(src_gray,[5,5], 5, src_gray, 5)
cv.fastNlMeansDenoising(src_gray,src_gray,75,7,21)
cv.fastNlMeansDenoising(src_gray,src_gray,25,7,21)
cv.fastNlMeansDenoising(src_gray,src_gray,25,7,21)
src_gray = cv.blur(src_gray, (3,3))
oi = src_gray
# Create Window
source_window = 'Source'
cv.namedWindow(source_window)
cv.imshow(source_window, original)
max_thresh = 255
thresh = 127 # initial threshold
cv.createTrackbar('Thresh:', source_window, thresh, max_thresh, thresh_callback)
thresh_callback(thresh)
cv.waitKey()


# import cv2
# import numpy as np
# image = cv2.imread('sample.png')

# img = image.copy()

# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# img = cv2.fastNlMeansDenoising(img,img,75,7,21)
# img = cv2.fastNlMeansDenoising(img,img,25,7,21)
# img = cv2.fastNlMeansDenoising(img,img,25,7,21)
# # cv2.imwrite('img_denoise.png', img)

# # cv2.GaussianBlur(img,[5,5], 5, img, 5)
# # cv2.imwrite('img_blur.png', img)

# temp, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
# # cv2.imwrite('img_thresh.png', thresh)

# # contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

# # img = cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB)

# # cv2.drawContours(img,contours,3, [0,255,0],1,cv2.LINE_AA,hierarchy)
# # cv2.imwrite("output.png",img)


# params = cv2.SimpleBlobDetector_Params()
# params.filterByArea = True
# params.minArea = 100


# ver = (cv2.__version__).split('.')
# if int(ver[0]) < 3 :
#     detector = cv2.SimpleBlobDetector(params)
# else :
#   detector = cv2.SimpleBlobDetector_create(params)

# keypoints = detector.detect(img)
# im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# cv2.imshow("Keypoints", im_with_keypoints)
# cv2.waitKey(0)

