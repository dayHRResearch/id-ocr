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

#ifndef OCR_INCLUDE_IMGPROC_CVTCOLOR_HPP
#define OCR_INCLUDE_IMGPROC_CVTCOLOR_HPP

#include "../log.hpp"

#include "opencv4/opencv2/imgproc.hpp"

using namespace cv;

/**
 * bgr to gray img func.
 * Args:
 *   image: Input color image
 *   imageGray: Converted grayscale image
 * Returns:
 *   success convert gray return 0, else return -1
 * @author: Changyu Liu
 * @last modify time: 2019.7.28
 */
int ConvertRGB2GRAY(const Mat &image, Mat &imageGray)
{
  if(!image.data||image.channels()!=3) {
    lprintf(MSG_ERROR, "image is empty or image channels not equal 3.\n");
    return -1;
  }
  imageGray=Mat::zeros(image.size(),CV_8UC1);
  uchar *pointImage=image.data;
  uchar *pointImageGray=imageGray.data;
  int stepImage=image.step;
  int stepImageGray=imageGray.step;
  for(int i=0;i<imageGray.rows;i++)
    for(int j=0;j<imageGray.cols;j++)
      pointImageGray[i*stepImageGray+j]=0.114*pointImage[i*stepImage+3*j]+0.587*pointImage[i*stepImage+3*j+1]+0.299*pointImage[i*stepImage+3*j+2];

  if (!imageGray.empty()) {
    lprintf(MSG_INFO, "BGR image convert Gray successful!\n");
    return 0;
  }
}

#endif //OCR_INCLUDE_IMGPROC_CVTCOLOR_HPP
