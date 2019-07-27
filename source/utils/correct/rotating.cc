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

#include "../../../include/correct/rotating.hpp"

using namespace cv;
using namespace std;

/**
 * print image rotating instructions.
 * @author Changyu Liu
 * @time 2019.7.28
 */
static void rotatingHelp()
{
  printf("\nThis program demonstrated the use of the rotating image.\n"
         "The rotating of an image is taken and it's power spectrum is displayed.\n");
}

/**
 * Image rotating.
 * Args:
 *   filename: The input image stream needed to perform the Fourier transform.
 * Returns:
 *   success transform return 0, else return -1.
 * @author Changyu Liu
 * @time 2019.7.28
 */
int rotating(const String &filename) {
  Mat image = imread(filename, IMREAD_COLOR);
  if (image.empty()) {
    lprintf(MSG_ERROR, "\tError open image!\n");
    return -1;
  }
  Mat grayImage;
  cvtColor(image, grayImage, COLOR_BGR2GRAY);

  const int nRows = grayImage.rows;
  const int nCols = grayImage.cols;
  // Compute the size of the Fourier transform
  int cRows = getOptimalDFTSize(nRows);
  int cCols = getOptimalDFTSize(nCols);
  Mat newImage;
  copyMakeBorder(grayImage, newImage, 0, cRows - nRows, 0, cCols - nCols, BORDER_CONSTANT, Scalar::all(0));
  if (!grayImage.empty())
    lprintf(MSG_INFO, "\tExpand image successful!\n");

  Mat groupMats[] = {Mat_<float>(newImage), Mat::zeros(newImage.size(), CV_32F) };
  Mat mergeMat;
  // Combine two pages into a 2-channel mat
  merge(groupMats, 2, mergeMat);
  // carry out discrete Fourier transform for the mat synthesized above,
  // which supports in-situ operation. The result of Fourier transform is complex.
  // Channel 1 stores the real part while channel 2 stores the imaginary part.
  dft(mergeMat, mergeMat);
  // The results of the transformation are divided into two pages of
  // each array to facilitate subsequent operations
  split(mergeMat, groupMats);
  // Find the amplitude of the Fourier change of each frequency,
  // put the amplitude on the first page
  magnitude(groupMats[0], groupMats[1], groupMats[0]);
  Mat magnitudeMat = groupMats[0].clone();
  magnitudeMat += Scalar::all(1);
  // The magnitude of the Fourier transform is so large that it doesn't fit on
  // the screen, where the higher values are white dots and the lower values
  // are black dots, changes of high and low values cannot be effectively
  // distinguished. In order to highlight the continuity of changes of high
  // and low values on the screen, we can replace the linear scale with
  // logarithmic scale
  log(magnitudeMat, magnitudeMat);
  // norm ops
  normalize(magnitudeMat, magnitudeMat, 0,1, NORM_MINMAX);
  magnitudeMat.convertTo(magnitudeMat, CV_8UC1, 255, 0);
  int cx = magnitudeMat.cols / 2;
  int cy = magnitudeMat.rows / 2;

  Mat tmp;
  Mat q0(magnitudeMat, Rect(0, 0, cx, cy));
  Mat q1(magnitudeMat, Rect(cx, 0, cx, cy));
  Mat q2(magnitudeMat, Rect(0, cy, cx, cy));
  Mat q3(magnitudeMat, Rect(cx, cy, cx, cy));

  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);

  q1.copyTo(tmp);
  q2.copyTo(q1);
  tmp.copyTo(q2);
  if (!magnitudeMat.empty())
    lprintf(MSG_INFO, "\tImage dft exchange successful!\n");

  Mat binaryMagnMat;
  threshold(magnitudeMat, binaryMagnMat, 155, 255, THRESH_BINARY);
  vector<Vec2f> lines;
  binaryMagnMat.convertTo(binaryMagnMat, CV_8UC1, 255, 0);
  HoughLines(binaryMagnMat, lines, 1, CV_PI / 180, 100, 0, 0);
  // cout << "lines.size:  " <<  lines.size() << endl;
  Mat houghMat(binaryMagnMat.size(), CV_8UC3);
  // Draw line
  // for (auto & i : lines) {
  //   float rho = i[0], theta = i[1];
  //   Point pt1, pt2;
  //   Coordinate transformation generates line expressions
  //   double a = cos(theta), b = sin(theta);
  //   double x0 = a*rho, y0 = b*rho;
  //   pt1.x = cvRound(x0 + 1000 * (-b));
  //   pt1.y = cvRound(y0 + 1000 * (a));
  //   pt2.x = cvRound(x0 - 1000 * (-b));
  //   pt2.y = cvRound(y0 - 1000 * (a));
  //   line(houghMat, pt1, pt2, Scalar(0, 0, 255), 1,8,0);
  // }
  double theta = 0.;
  //检测线角度判断
  for (auto & line : lines) {
    double thetaTemp = line[1] * 180 / CV_PI;
    if (thetaTemp > 0 && thetaTemp < 90)
    {
      theta = thetaTemp;
      break;
    }
  }
  // angle
  double angelT = nRows * tan(theta / 180 * CV_PI) / nCols;
  theta = (atan(angelT)) * 180 / CV_PI;
  cout << "theta: " << theta << endl;

  // get image center
  Point2f centerPoint = Point2f(nCols / 2, nRows / 2);
  double scale = 1;
  Mat warpMat = getRotationMatrix2D(centerPoint, theta, scale);
  Mat resultImage(grayImage.size(), grayImage.type());
  warpAffine(grayImage, resultImage, warpMat, resultImage.size());
  if (!resultImage.empty())
    lprintf(MSG_INFO, "\tImage rotating successful!\n");
  imwrite(filename, resultImage);
  return 0;
}
