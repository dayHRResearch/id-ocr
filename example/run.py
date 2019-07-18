import os

import cv2
from skimage import io, transform
import numpy as np


def load_detector():
    face = cv2.CascadeClassifier('../data/face.xml')
    eye = cv2.CascadeClassifier('../data/eye.xml')

    return face, eye


def check_exists():
    # Create a directory if the file directory does not exist.
    if not os.path.exists('./info/face'):
        os.makedirs('./info/face')


#
# 计算图像的身份证倾斜的角度
def IDcorner(landmarks):
    """landmarks:检测的人脸5个特征点
       经过测试使用第0个和第2个特征点计算角度较合适
    """
    corner20 = twopointcor(landmarks[2, :], landmarks[0, :])
    corner = np.mean([corner20])
    return corner


# 计算眼睛的倾斜角度,逆时针角度
def twopointcor(point1, point2):
    """point1 = (x1,y1),point2 = (x2,y2)"""
    deltxy = point2 - point1
    corner = np.arctan(deltxy[1] / deltxy[0]) * 180 / np.pi
    return corner


def correct_image(image_path):
    """ Fixed left to right tilt of photo.

    Args:
        image_path: The image to be processed.

    Returns:

    """
    image = io.imread(image_path)
    image = image[:, :, :: -1]
    # image_name = os.path.splitext(os.path.split(image_path)[1])[0]
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    eyes = eye_detector.detectMultiScale(
        image,
        scaleFactor=1.3,
        minNeighbors=9,
        minSize=(55, 55),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in eyes:
        # eye = image[y-50:y + h+60, x-35:x + w+35]

        landmarks = [x, y, w, h]
        corner = IDcorner(landmarks)
        image2 = transform.rotate(image, corner, clip=False)
        image2 = np.uint8(image2 * 255)



    # eye_detector = dlib.get_frontal_face_detector()
    # dets = detector(image, 2) # 使用detector进行人脸检测 dets为返回的结果
    # 检测人脸的眼睛所在位置
    # predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
    # detected_landmarks = predictor(image, dets[0]).parts()
    # landmarks = np.array([[p.x, p.y] for p in detected_landmarks])
    # corner = IDcorner(landmarks)
    # ## 旋转后的图像
    # image2 = transform.rotate(image,corner,clip=False)
    # image2 = np.uint8(image2*255)
    # ## 旋转后人脸位置
    # det = detector(image2, 2)
    # return image2,det


def face_detector(image_path):
    image = io.imread(image_path)
    # image = image[:, :, :: -1]
    image_name = os.path.splitext(os.path.split(image_path)[1])[0]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=9,
        minSize=(55, 55),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        face = image[y - 50:y + h + 60, x - 35:x + w + 35]

        return image_name, face


def save_face(filename, face):
    """ Detect and save the face on the id.

    Args:
        filename:
        face:

    """

    # bgr to rgb
    cv2.imwrite('./info/face/' + filename + '_face.png', face[:, :, :: -1])
    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)


if __name__ == '__main__':
    check_exists()
    face_detector, eye_detector = load_detector()
    correct_image('images/demo1.png')
    # save_face()
