import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('mask_oil.png', 0)



img = cv2.medianBlur(img, 13)

h, w = img.shape
for i in range(h-1):
    for j in range(w-1):
        if img[i][j] < 100:
            img[i][j] = 0
        else:
            pass

cv2.imshow('img', img)
img = cv2.adaptiveThreshold(img, 255,
                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY, 11, 2)
#img = cv2.GaussianBlur(img, (7, 7), 0)

contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
dst = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.imwrite('mask_dst2.jpg', dst)
dst = cv2.drawContours(dst, contours, -1, (0, 255, 0), 3)


cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()