import pytesseract
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import io
import numpy as np
import pandas as pd
import re

image = io.imread("./images/w1.jpg")
print(image.shape)
high, width = image.shape[0:2]
# 图像放大两倍 ，4x4像素邻域的双三次插值
image = cv2.resize(image, (width * 4, high * 4), interpolation=cv2.INTER_CUBIC)


imagegray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
retval, imagebin = cv2.threshold(
    imagegray, 120, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

text = pytesseract.image_to_string(imagebin, lang='chi_sim')
print(text)

# 提取识别的内容
textlist = text.split("\n")
textdf = pd.DataFrame({"text": textlist})
textdf["textlen"] = textdf.text.apply(len)
# 去除长度《＝1的行
textdf = textdf[textdf.textlen > 1].reset_index(drop=True)

# 提取相应的信息
print("第一行:", textdf.text[0][-7:])
print("=====================")
print("第二行:", textdf.text[1][-5:])
print("=====================")
print("签发机关:", textdf.text[2].split(" ")[-1])
print("=====================")
# 提取时间信息
timetext = textdf.text[3]
timetext = re.findall(r"\d+", timetext)
print("有效开始时间:", ".".join(timetext[0:3]))
print("=====================")
print("失效时间:", ".".join(timetext[3:]))
print("=====================")
