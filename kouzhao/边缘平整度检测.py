import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('mask_oil.png', 0)

print(img.shape)

img = cv2.medianBlur(img, 13)
ret, dst = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

dst = cv2.GaussianBlur(dst, (7, 7), 0)

k = np.ones((21, 21), np.uint8)
dst2 = cv2.morphologyEx(dst, cv2.MORPH_OPEN, k)


#cv2.imshow('dst', dst)
cannydst = cv2.Canny(dst2, 50, 255, cv2.THRESH_BINARY)
cv2.imshow('cannydst', cannydst)

y, x = cannydst.shape
print(x, y)

point = []

for i in range(x):
    for j in range(y):
        if cannydst[j][i] == 255:
            point.append([i, j])
        else:
            pass
print(point[1][0])

y = int(y/2)

top_half_point = []
for i in range(len(point)):
    if point[i][1] > y:
        top_half_point.append(point[i])
    else:
        pass

#获取上半部分的横坐标
top_half_point_x = []

for i in range(len(top_half_point)):
    top_half_point_x.append(top_half_point[i][0])
#获取上半部分的横坐标的最大最小值
x_min = min(top_half_point_x)
x_max = max(top_half_point_x)

print(x_max, x_min)
#去除小部分的左右边界，精确提取上边界
X = img.shape[1]
Y = img.shape[0]
#上边界左右边界修正值（调参）

a = int(X/15)

X_min = x_min + a
X_max = x_max - a

#提取上边界需要计算的y坐标
top_half_point_y = []
for i in range(len(top_half_point)):
    if top_half_point[i][0]>=X_min and top_half_point[i][0]<=X_max:
        top_half_point_y.append(top_half_point[i][1])
    else:
        pass
print(top_half_point_y)

#算出均值，并用方差来表示点的波动程度
y_mean = np.mean(top_half_point_y)
print(y_mean)

#求出方差，设定阈值来判断平整与否
y_arr = np.var(top_half_point_y)
print(y_arr)

cv2.waitKey(0)
cv2.destroyAllWindows()






