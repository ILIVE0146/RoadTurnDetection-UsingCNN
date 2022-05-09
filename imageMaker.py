from __future__ import print_function
from glob import glob
from importlib.resources import path
import os
import cv2
from cv2 import imshow
from cv2 import blur
import numpy as np

def thresholdImg(inputPath, outputPath, thresholdVal = 127):
    img_mask = inputPath
    img_names = glob(img_mask)
    if outputPath[-1] == '/' or outputPath[-1] == '\\':
        outputPath = outputPath[:-1]

    for fn in img_names:
        im_gray = cv2.imread(fn, 0)
        t, im = cv2.threshold(im_gray, thresholdVal, 255, cv2.THRESH_BINARY)
        name = os.path.basename(fn)
        outfile = outputPath + '/t-' + name
        cv2.imwrite(outfile, im)


def flipImages(ipath, opath):

    img_mask = ipath
    img_names = glob(img_mask)

    if opath[-1] == '/' or opath[-1] == '\\':
        opath = opath[:-1]

    for fn in img_names:
        im_gray = cv2.imread(fn, 0)
        im_flip = cv2.flip(im_gray, 1)

        name = os.path.basename(fn)
        outfile = opath + '/f-' + name
        cv2.imwrite(outfile, im_flip)




def blurImages(ipath):
    img_mask = ipath
    img_names = glob(img_mask)
    thresholdValue = 80
    for fn in img_names:
        src = cv2.imread(fn)
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        cv2.GaussianBlur(src_gray,[5,5], 5, src_gray, 5)
        t, src_gray = cv2.threshold(src_gray, thresholdValue, 255, cv2.THRESH_BINARY_INV)
        cv2.fastNlMeansDenoising(src_gray,src_gray,75,7,21)
        cv2.fastNlMeansDenoising(src_gray,src_gray,25,7,21)
        cv2.fastNlMeansDenoising(src_gray,src_gray,25,7,21)
        src_gray = cv2.blur(src_gray, (3,3))
        t, src_gray = cv2.threshold(src_gray, thresholdValue, 255, cv2.THRESH_BINARY_INV)
        cv2.GaussianBlur(src_gray,[15,15], 5, src_gray, 5)
        src_gray = cv2.blur(src_gray, (3,3))
        t, src_gray = cv2.threshold(src_gray, thresholdValue, 255, cv2.THRESH_BINARY)

        name = os.path.basename(fn)
        cv2.imwrite("./t-" + name,src_gray)


if __name__ == "__main__":
    blurImages("original.jpg")
