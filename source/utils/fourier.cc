/**
 * Copyright 2019 DayHR Authors. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================
 */

#include "../../include/fourier.hpp"

using namespace cv;
using namespace std;

/**
 * print discrete Fourier transform instructions
 * @author Changyu Liu
 * @time 2019.7.27
 */
static void dftHelp() {
  printf("\nThis program demonstrated the use of the discrete Fourier transform (DFT).\n"
         "The dft of an image is taken and it's power spectrum is displayed.\n");
}

/**
 * discrete Fourier transform (DFT).
 * Args:
 *   filename: The input image stream needed to perform the Fourier transform.
 * Returns:
 *   success transform return 0, else return -1.
 * @author Changyu Liu
 * @time 2019.7.27
 */
int fourierTransform(const String &filename) {
  Mat image = imread(filename, IMREAD_COLOR);
  if (image.empty()) {
    lprintf(MSG_ERROR, "\tError opening image!\n");
    return -1;
  }
  Mat gray_image;
  cvtColor(image, gray_image, IMREAD_GRAYSCALE);
  if (!gray_image.empty())
    lprintf(MSG_INFO, "\tConvert gray scale image successful!\n");

  // step 1: Expand input gray_image to optimal size.
  Mat padded;
  int height = getOptimalDFTSize(gray_image.rows);
  int width = getOptimalDFTSize(gray_image.cols); // on the border add zero values
  copyMakeBorder(gray_image, padded, 0, height - gray_image.rows, 0, width - gray_image.cols, BORDER_CONSTANT,
                 Scalar::all(0));
  if (!padded.empty())
    lprintf(MSG_INFO, "\tExpand gray_image to optimal size successful!\n");

  // step2: Add to the expanded another plane with zeros.
  Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
  Mat complexImage;
  merge(planes, 2, complexImage);
  if (!complexImage.empty())
    lprintf(MSG_INFO, "\tAdd to the expanded another plane with zeros successful!\n");

  // step3: this way the result may fit in the source matrix
  dft(complexImage, complexImage);
  if (!complexImage.empty())
    lprintf(MSG_INFO, "\tCal dft successful!\n");

  // step4: compute the magnitude and switch to logarithmic scale
  split(complexImage, planes);                // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
  magnitude(planes[0], planes[1], planes[0]); // planes[0] = magnitude
  Mat magImage = planes[0];
  if (!magImage.empty())
    lprintf(MSG_INFO, "\tcompute the magnitude and switch to logarithmic scale successful!\n");

  // step 5: switch to logarithmic scale
  magImage += Scalar::all(1);
  // Calculates x and y coordinates of 2D vectors from their magnitude and angle.
  log(magImage, magImage);
  if (!magImage.empty())
    lprintf(MSG_INFO, "\tswitch to logarithmic scale successful!\n");

  // step 6: crop rearrange
  // crop the spectrum, if it has an odd number of rows or columns
  magImage = magImage(Rect(0, 0, magImage.cols & -2, magImage.rows & -2));
  // rearrange the quadrants of Fourier gray_image  so that the origin is at the gray_image center
  int cx = magImage.cols / 2;
  int cy = magImage.rows / 2;
  // Create a ROI per quadrant
  Mat q0(magImage, Rect(0, 0, cx, cy));   // Top-Left
  Mat q1(magImage, Rect(cx, 0, cx, cy));  // Top-Right
  Mat q2(magImage, Rect(0, cy, cx, cy));  // Bottom-Left
  Mat q3(magImage, Rect(cx, cy, cx, cy)); // Bottom-Right
  // swap quadrants (Top-Left with Bottom-Right)
  Mat tmp;
  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);
  // swap quadrant (Top-Right with Bottom-Left)
  q1.copyTo(tmp);
  q2.copyTo(q1);
  tmp.copyTo(q2);
  if (!magImage.empty())
    lprintf(MSG_INFO, "\tcrop rearrange successful!\n");

  // step7: normalize gray_image
  normalize(magImage, magImage, 0, 1, NORM_MINMAX);
  if (!magImage.empty())
    lprintf(MSG_INFO, "\tImage normalize successful!\n");

  Mat magImg(magImage.size(), CV_8UC1);
  magImage.convertTo(magImg, CV_8UC1, 255, 0);
  imwrite(filename, magImage);
  lprintf(MSG_INFO, "\tImage discrete Fourier transform successful!\n");

  return 0;
}
