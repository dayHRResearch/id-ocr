#include <iostream>
#include "opencv4/opencv2/highgui.hpp"
#include "opencv4/opencv2/imgproc.hpp"

using namespace cv;
using namespace std;

//第一个参数：输入图片名称；第二个参数：输出图片名称
void GetContoursPic(const char* pSrcFileName, const char* pDstFileName) {
  Mat srcImg = imread(pSrcFileName);
  imshow("原始图", srcImg);
  Mat gray, binImg;
  //灰度化
  cvtColor(srcImg, gray, COLOR_RGB2GRAY);
  imshow("灰度图", gray);
  //二值化
  threshold(gray, binImg, 100, 200, THRESH_BINARY);
  imshow("二值化", binImg);

  vector<vector<Point> > contours;
  vector<Rect> boundRect(contours.size());
  //注意第5个参数为CV_RETR_EXTERNAL，只检索外框
  findContours(binImg, contours, RETR_EXTERNAL,
               CHAIN_APPROX_NONE);  //找轮廓
  cout << contours.size() << endl;
  for (int i = 0; i < contours.size(); i++) {
    //需要获取的坐标
    Point2d rectpoint[4];
    Mat rect = minAreaRect(Mat(contours[i]));

    cvBoxPoints(rect, rectpoint);  //获取4个顶点坐标
    //与水平线的角度
    float angle = rect.angle;
    cout << angle << endl;

    int line1 = sqrt(
        (rectpoint[1].y - rectpoint[0].y) * (rectpoint[1].y - rectpoint[0].y) +
        (rectpoint[1].x - rectpoint[0].x) * (rectpoint[1].x - rectpoint[0].x));
    int line2 = sqrt(
        (rectpoint[3].y - rectpoint[0].y) * (rectpoint[3].y - rectpoint[0].y) +
        (rectpoint[3].x - rectpoint[0].x) * (rectpoint[3].x - rectpoint[0].x));
    // rectangle(binImg, rectpoint[0], rectpoint[3], Scalar(255), 2);
    //面积太小的直接pass
    if (line1 * line2 < 600) {
      continue;
    }

    //为了让正方形横着放，所以旋转角度是不一样的。竖放的，给他加90度，翻过来
    if (line1 > line2) {
      angle = 90 + angle;
    }

    //新建一个感兴趣的区域图，大小跟原图一样大
    Mat RoiSrcImg(srcImg.rows, srcImg.cols, CV_8UC3);  //注意这里必须选CV_8UC3
    RoiSrcImg.setTo(0);  //颜色都设置为黑色
    // imshow("新建的ROI", RoiSrcImg);
    //对得到的轮廓填充一下
    drawContours(binImg, contours, -1, Scalar(255), FILLED);

    //抠图到RoiSrcImg
    srcImg.copyTo(RoiSrcImg, binImg);

    //再显示一下看看，除了感兴趣的区域，其他部分都是黑色的了
    namedWindow("RoiSrcImg", 1);
    imshow("RoiSrcImg", RoiSrcImg);

    //创建一个旋转后的图像
    Mat RatationedImg(RoiSrcImg.rows, RoiSrcImg.cols, CV_8UC1);
    RatationedImg.setTo(0);
    //对RoiSrcImg进行旋转
    Point2f center = rect.center;                    //中心点
    Mat M2 = getRotationMatrix2D(center, angle, 1);  //计算旋转加缩放的变换矩阵
    warpAffine(RoiSrcImg, RatationedImg, M2, RoiSrcImg.size(), 1, 0,
               Scalar(0));  //仿射变换
    imshow("旋转之后", RatationedImg);
    imwrite("r.jpg", RatationedImg);  //将矫正后的图片保存下来
  }

#if 1
  //对ROI区域进行抠图

  //对旋转后的图片进行轮廓提取
  vector<vector<Point> > contours2;
  Mat raw = imread("r.jpg");
  Mat SecondFindImg;
  // SecondFindImg.setTo(0);
  cvtColor(raw, SecondFindImg, COLOR_BGR2GRAY);  //灰度化
  threshold(SecondFindImg, SecondFindImg, 80, 200, CV_THRESH_BINARY);
  findContours(SecondFindImg, contours2, RETR_EXTERNAL, CHAIN_APPROX_NONE);
  // cout << "sec contour:" << contours2.size() << endl;

  for (int j = 0; j < contours2.size(); j++) {
    //这时候其实就是一个长方形了，所以获取rect
    Rect rect = boundingRect(Mat(contours2[j]));
    //面积太小的轮廓直接pass,通过设置过滤面积大小，可以保证只拿到外框
    if (rect.area() < 600) {
      continue;
    }
    Mat dstImg = raw(rect);
    imshow("dst", dstImg);
    imwrite(pDstFileName, dstImg);
  }
#endif
}

int main() {
  GetContoursPic("./images/demo2.jpg", "FinalImage.jpg");
  waitKey();
  return 0;
}