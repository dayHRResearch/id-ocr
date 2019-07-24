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


# import the necessary packages
from imutils.perspective import four_point_transform
import cv2


def main():
    # Gray scale processing of BGR images.
    image = cv2.imread("sfz_back1.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blurs an image using a Gaussian filter.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Finds edges in an image using the Canny algorithm.
    edged = cv2.Canny(blurred, 75, 200)
    # Finds contours in a binary image.
    contours, _ = cv2.findContours(
        edged.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    doc_cnt = None

    if len(contours) > 0:
        # Sort the size of the rectangular box found.
        new_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for c in new_contours:
            # Calculates a contour perimeter or a curve length.
            peri = cv2.arcLength(c, True)
            # Approximates a polygonal curve(s) with the specified precision.
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # If four fixed points are found.
            if len(approx) == 4:
                doc_cnt = approx
                break

    result_image = four_point_transform(image, doc_cnt.reshape(4, 2))
    cv2.imwrite("result.png", result_image)


if __name__ == '__main__':
    main()
