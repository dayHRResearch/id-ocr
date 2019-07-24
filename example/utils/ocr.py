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

import cv2
import dlib
import numpy as np
import pandas as pd
import pytesseract
from skimage import io, transform


def cal_binocular_position(point1, point2):
    """ Calculate the angle of the human eye tilt in the picture.

    Args:
        point1: Left eye coordinates.
        point2: Right eye coordinates.

    Returns:
        Human Eye Migration Angle Size.

    `point1 = (x1,y1), point2 = (x2,y2)`

    """
    distance = point2 - point1
    angle = np.arctan(distance[1] / distance[0]) * 180 / np.pi
    return angle


def cal_inclination_angle(landmarks):
    """ Calculate the tilt angle of the ID card of the image.
    
    Args:
        landmarks: Five Feature Points of Face Detection.

    Returns:
        The Angle of Inclination of Identity Card.

    """
    angle = cal_binocular_position(landmarks[2, :], landmarks[0, :])
    angle = np.mean([angle])
    return angle


def rotate_card(image):
    """ Correct the photo.

    Args:
        image: Image to be processed.

    Returns:

    """
    # 使用dlib.get_frontal_face_detector识别人脸
    detector = dlib.get_frontal_face_detector()
    dets = detector(image, 2)  # 使用detector进行人脸检测 dets为返回的结果
    # 检测人脸的眼睛所在位置
    predictor = dlib.shape_predictor("../../data/shape_predictor_5_face_landmarks.dat")
    detected_landmarks = predictor(image, dets[0]).parts()
    landmarks = np.array([[p.x, p.y] for p in detected_landmarks])
    corner = cal_inclination_angle(landmarks)
    # 旋转后的图像
    image2 = transform.rotate(image, float(corner), clip=False)
    image2 = np.uint8(image2 * 255)
    # 旋转后人脸位置
    det = detector(image2, 2)
    return image2, det


def main(image, threshod=120):
    # 转正身份证：
    image2, dets = rotate_card(image)
    # 提取照片的头像
    # 在图片中标注人脸，并显示
    left = dets[0].left()
    top = dets[0].top()
    right = dets[0].right()
    bottom = dets[0].bottom()
    # 照片的位置（不怎么精确）
    width = right - left
    high = top - bottom
    left2 = np.uint(left - 0.3 * width)
    bottom2 = np.uint(bottom + 0.5 * width)
    # 身份证上人的照片
    top2 = np.uint(bottom2 + 2.05 * high)
    right2 = np.uint(left2 + 1.7 * width)
    rectangle = [(left2, bottom2), (top2, right2)]
    imageperson = image2[top2:bottom2, left2:right2, :]
    imageperson = cv2.cvtColor(imageperson, cv2.COLOR_BGR2RGB)
    cv2.imwrite("a.png", imageperson)
    # 对图像进行处理，转化为灰度图像=>二值图像
    imagegray = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    retval, imagebin = cv2.threshold(
        imagegray, threshod, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    # 将照片去除
    imagebin[0:bottom2, left2:-1] = 255
    # 通过pytesseract库来查看检测效果，但是结果并不是很好
    text = pytesseract.image_to_string(imagebin, lang='chi_sim')
    textlist = text.split("\n")
    textdf = pd.DataFrame({"text": textlist})
    textdf["textlen"] = textdf.text.apply(len)
    # 去除长度《＝1的行
    textdf = textdf[textdf.textlen > 1].reset_index(drop=True)
    return image2, dets, rectangle, imagebin, textdf


# 识别身份证的信息
image = io.imread("./images/sfz_back2.png")
image2, dets, rectangle, imagebin, textdf = main(image, threshod=120)

print(textdf)
# # 提取相应的信息
# print("姓名:", textdf.text[0])
# print("=====================")
# print("性别:", textdf.text[1].split(" ")[0])
# print("=====================")
# print("民族:", textdf.text[1].split(" ")[-1])
# print("=====================")
# yearnum = textdf.text[2].split(" ")[0]  # 提取数字
# yearnum = re.findall(r"\d+", yearnum)[0]
# print("出生年:", yearnum)
# print("=====================")
# monthnum = textdf.text[2].split(" ")[1]  # 提取数字
# monthnum = re.findall(r"\d+", monthnum)[0]
# print("出生月:", monthnum)
# print("=====================")
# daynum = textdf.text[2].split(" ")[2]  # 提取数字
# daynum = re.findall(r"\d+", daynum)[0]
# print("出生日:", daynum)
# print("=====================")
IDnum = textdf.text.values[-1]
if len(IDnum) > 18:  # 去除不必要的空格
    IDnum = IDnum.replace(" ", "")
print("公民身份证号:", IDnum)
# print("=====================")
# # 获取地址，因为地址可能会是多行
# desstext = textdf.text.values[3:(textdf.shape[0] - 1)]
# print("地址:", "".join(desstext))
# print("=====================")
