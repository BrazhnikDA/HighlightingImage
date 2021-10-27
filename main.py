import sys

import cv2 as cv
import numpy as np


def CalcOfDamageAndNonDamage(image, name):
    hsv_img = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    markers = np.zeros((image.shape[0], image.shape[1]), dtype="int32")

    markers[90: 140, 90: 140] = 255

    markers[236: 255, 0: 20] = 1

    markers[0: 20, 0: 20] = 1

    markers[0: 20, 236: 255] = 1

    markers[236: 255, 236: 255] = 1

    leafs_area_BGR = cv.watershed(image, markers)

    healthy_part = cv.inRange(hsv_img, (36, 25, 25), (86, 255, 255))

    ill_part = leafs_area_BGR - healthy_part

    mask = np.zeros_like(image, np.uint8)
    mask[leafs_area_BGR > 1] = (0, 255, 100)
    mask[ill_part > 1] = (0, 0, 255)

    cv.imshow("hsv_img_for {}".format(name), hsv_img)

    return mask


# @parameters - image, sizeMatrix
# image - input image
# sizeMatrix - size of Matrix for filter
# @return - End of Matrix after apply filter
def GaussianFilter(image, sizeMatrix):
    resImage = cv.GaussianBlur(image, (sizeMatrix, sizeMatrix), cv.BORDER_DEFAULT)
    cv.imshow("Gaussian Filter", resImage)

    return resImage


# @parameters - image, sizeMatrix
# image - input image
# sizeMatrix - size of Matrix for filter
# @return - End of Matrix after apply filter
def ErodedFilter(image, sizeMatrix):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (sizeMatrix, sizeMatrix))
    resErode = cv.erode(image, kernel)
    cv.imshow("Eroded Filter", resErode)

    return resErode


# @parameters - image, sizeMatrix
# image - input image
# sizeMatrix - size of Matrix for filter
# @return - End of Matrix after apply filter
def MedianFilter(image, sizeMatrix):
    imgBlur = cv.medianBlur(image, sizeMatrix, sizeMatrix)
    cv.imshow("Median Filter", imgBlur)

    return imgBlur


def main():
    # Constant
    NAME_KEY_ORIGINAL = "Original"
    NAME_KEY_ERODED = "Eroded"
    NAME_KEY_GAUSSIAN = "Gaussian"
    NAME_KEY_MEDIAN = "Median"

    # Size matrix for filter
    sizeMatrix = 7

    img = cv.imread('C:\\Users\\DIMA\\Downloads\\data\\3.jpg')
    if img is None:
        sys.exit("Could not read the image.")

    (b, g, r) = img[0, 0]
    print("Red: {}, Green: {}, Blue: {}".format(r, g, b))
    cv.imshow("Orig", img)

    cv.imshow("1" + NAME_KEY_ORIGINAL, CalcOfDamageAndNonDamage(img, NAME_KEY_ORIGINAL))

    gaussianFilter = GaussianFilter(img, sizeMatrix)
    resImg = CalcOfDamageAndNonDamage(gaussianFilter, NAME_KEY_GAUSSIAN)
    cv.imshow("Result {}".format(NAME_KEY_GAUSSIAN), resImg)

    erodedFilter = ErodedFilter(img, sizeMatrix)
    resImg = CalcOfDamageAndNonDamage(erodedFilter, NAME_KEY_ERODED)
    cv.imshow("Result {}".format(NAME_KEY_ERODED), resImg)

    medianFilter = MedianFilter(img, sizeMatrix)
    resImg = CalcOfDamageAndNonDamage(medianFilter, NAME_KEY_MEDIAN)
    cv.imshow("Result {}".format(NAME_KEY_MEDIAN), resImg)

    k = cv.waitKey(0)


if __name__ == '__main__':
    main()
