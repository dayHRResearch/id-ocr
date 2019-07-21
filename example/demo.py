import cv2

img = cv2.imread('./images/demo1.png')
GrayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

CannyImage = cv2.Canny(GrayImage, 50, 150, apertureSize=3)

ret, BinImage = cv2.threshold(CannyImage, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(
    BinImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.imshow('bin', BinImage)
cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
cv2.imshow('edge', img)
cv2.waitKey(0)
