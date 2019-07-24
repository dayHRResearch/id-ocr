# Copyright 2019 ChangyuLiu Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import time

import cv2
import numpy as np


def morphological_transformation(input_dir):
    """ Performs advanced morphological transformations.

    Args:
        input_dir: Input Picture Data Stream.

    Returns:
        Picture Data Stream after Rotation Correction Processing.

    """
    raw_image = cv2.imread(input_dir)
    gray_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)

    # Gauss Fuzzy De-noising (Setting the Size of Convolution Kernel Affects
    # the Effect).
    blur_image = cv2.GaussianBlur(gray_image, (9, 9), 0)
    # Setting threshold 165 (Threshold affects open-close operation effect).
    _, threshold = cv2.threshold(blur_image, 165, 255, cv2.THRESH_BINARY)
    # Define rectangular structural elements.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # Closed operation (link block)
    closed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
    # # Open Operations (De-noising Points)
    image = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    return raw_image, image


def correct_image(raw_image, image):
    """ Image Correction Processing Function.

    Args:
        raw_image: Image streams that need to be tailored.
        image: Picture streams after cropping.

    Returns:
        Correct picture flow.

    """
    # Finds contours in a binary image.
    contours, hierarchy = cv2.findContours(
        image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Rotating bounding box for calculating maximum contour
    max_contour = sorted(contours, key=cv2.contourArea, reverse=True)[1]
    # Computational Minimum Matrix Frame
    min_area_rect = cv2.minAreaRect(max_contour)

    rows, cols = raw_image.shape[:2]

    # Calculates an affine matrix of 2D rotation.
    image = cv2.getRotationMatrix2D((cols / 2, rows / 2), min_area_rect[2], 1)
    # Applies an affine transformation to an image.
    correct_image = cv2.warpAffine(raw_image, image, (cols, rows))

    return correct_image


def findContours_img(original_img, opened):
    contours, hierarchy = cv2.findContours(opened, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(contours, key=cv2.contourArea, reverse=True)[1]          # 计算最大轮廓的旋转包围盒
    rect = cv2.minAreaRect(c)
    angle = rect[2]
    print("angle",angle)
    box = np.int0(cv2.boxPoints(rect))
    draw_img = cv2.drawContours(original_img.copy(), [box], -1, (0, 0, 255), 3)
    rows, cols = original_img.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    result_img = cv2.warpAffine(original_img, M, (cols, rows))
    return result_img,draw_img


if __name__ == "__main__":
    start = time.time()
    input_dir = "./demo2.png"

    raw_image, image = morphological_transformation(input_dir)

    correct_image = correct_image(raw_image, image)
    result_img, draw_img = findContours_img(raw_image, image)

    cv2.imwrite("result.png", correct_image)
    cv2.imshow("d", draw_img)
    cv2.waitKey()

    print(f"Correct images time use {time.time() - start:.4f} s!")
