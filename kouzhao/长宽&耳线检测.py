import numpy as np
import cv2

#对旋转前的图像进行开运算，提取除了耳线的口罩矩形区域

img = cv2.imread('mask_dst2.jpg', cv2.IMREAD_UNCHANGED)
k = np.ones((19, 19), np.uint8)
result = cv2.morphologyEx(img, cv2.MORPH_OPEN, k)

image = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
image = cv2.GaussianBlur(image, (7, 7), 0)
cv2.imshow('image', image)
cv2.waitKey(0)

'''
x, y, w, h = cv2.boundingRect(image)
img = cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)  #不具备旋转角度的矩形框架
'''

#提取口罩除耳线的外接最小矩形
contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP,
                                       cv2.CHAIN_APPROX_TC89_L1)
result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
img3 = cv2.drawContours(result, contours, -1, (0, 255, 0), 3)
cv2.imshow('img3', img3)
cv2.waitKey(0)
#print(contours)
rect = cv2.minAreaRect(contours[0])
#rect_a = np.array(rect)
print(rect[2])  #最小外接矩形的偏移旋转角度
box = cv2.boxPoints(rect)
box = np.int0(box)
print(box)
result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#绘制旋转回正前的外接最小矩形
img = cv2.drawContours(result, [box], 0, (0, 0, 255), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
#对图像进行旋转回正
x = img.shape[1]
y = img.shape[0]
#获取旋转后的图像
h = rect[0][1]
w = rect[0][0]
cx = w//2
cy = h//2
M = cv2.getRotationMatrix2D((cx, cy), rect[2], 1.0)
cos = np.abs(M[0, 0])
sin = np.abs(M[0, 1])
nW = int((h*sin) + (w*cos))
nH = int((h*cos) + (w*sin))

M[0, 2] += (nW/2) - cx
M[1, 2] += (nH/2) - cy
print("M:", M)
image2 = cv2.imread('mask_dst2.jpg', cv2.IMREAD_UNCHANGED)

#获取除去耳线前的旋转回正总图像
rotate_mask = cv2.warpAffine(image2, M, (x, y))
cv2.imshow('rotate_mask', rotate_mask)
#再次进行开运算，提取旋转回正之后的除耳线的口罩矩形区域
k = np.ones((17, 17), np.uint8)
result_rotate_mask = cv2.morphologyEx(rotate_mask, cv2.MORPH_OPEN, k)
#cv2.imshow('ss', result_rotate_mask)
result_rotate_mask = cv2.cvtColor(result_rotate_mask, cv2.COLOR_BGR2GRAY)####

contours2, hierarchy2 = cv2.findContours(result_rotate_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

rect_result_rotate_mask = cv2.minAreaRect(contours2[0])#获取框选矩形的坐标，长宽

print(rect_result_rotate_mask) #获取口罩的长宽与中心点坐标
box = cv2.boxPoints(rect_result_rotate_mask)
box = np.int0(box)


result_rotate_mask = cv2.cvtColor(result_rotate_mask, cv2.COLOR_GRAY2BGR)

result_rotate_mask = cv2.drawContours(result_rotate_mask, [box], 0, (0, 0, 255), 2)

#cv2.imshow('result_rotate_mask', result_rotate_mask)
#cv2.waitKey(0)
#确定box的四个顶点
x_min = np.min(box[:, 0])
x_max = np.max(box[:, 0])
y_min = np.min(box[:, 1])
y_max = np.max(box[:, 1])
box_coordinates = np.array([[x_min, y_min], [x_max, y_min],
                            [x_min, y_max], [x_max, y_max]])

#获取口罩的四个端点位置为耳线位置的定位提供基准
print(box_coordinates)


#通过对旋转后的图像的加减提取耳线
subtracted = cv2.subtract(rotate_mask, result_rotate_mask)

k2 = np.ones((5, 5), np.uint8)
result_subtracted = cv2.morphologyEx(subtracted, cv2.MORPH_OPEN, k2)
result_subtracted = cv2.GaussianBlur(result_subtracted, (3, 3), 0) #回正后的耳线图

result_subtracted = cv2.cvtColor(result_subtracted, cv2.COLOR_BGR2GRAY)

cv2.imshow('result_subtracted', result_subtracted) #返回回正之后的耳线图

#提取耳线上的坐标点并对耳线上的坐标点分析与最小外接矩形相交的点的坐标，算出与边界的像素点。


height, width = result_subtracted.shape
#print(height, width)

points = []
for i in range(height-1):
    for j in range(width-1):
        if result_subtracted[i][j] != 0:
            a = [i, j]
            points.append(a)
        else:
            pass
print(points)  #(y, x)
#print(len(points))
points = np.array(points)

#box_coordinates为矩形的四个顶点
left_box_coordinate = box_coordinates[0][0]
right_box_coordinate = box_coordinates[1][0]
high_box_coordinate = box_coordinates[0][1]
low_box_coordinate = box_coordinates[2][1]
mid_high = (low_box_coordinate-high_box_coordinate)/2+high_box_coordinate
print(mid_high)

#取上半部分的耳线点
points_top_half = []
for i in range(len(points)):
    if points[i][0] <= mid_high:
        points_top_half.append(points[i])
    else:
        pass

points_top_half = np.array(points_top_half)
points_top_half_x = []
for i in range(len(points_top_half)):
    points_top_half_x.append(points_top_half[i][1])
#print(points_top_half)
#取下半部分的耳线点
points_low_half = []
for i in range(len(points)):
    if points[i][0] >= mid_high:
        points_low_half.append(points[i])
    else:
        pass
points_low_half = np.array(points_low_half)
points_low_half_x = []
#取下半部分的x坐标
for i in range(len(points_low_half)):
    points_low_half_x.append(points_low_half[i][1])
#print(points_low_half)

'''
#取耳线轮廓点的横坐标（即[i][1]）
points_x = []
for i in range(len(points)):
    points_x.append(points[i][1])

print(points_x)
'''
#取耳线离矩形区域最近的点
#定义距离函数
def find_close(arr, e):

    size = len(arr)
    idx = 0
    val = abs(e - arr[idx])

    for i in range(1, size):
        val1 = abs(e - arr[i])
        if val1 < val:
            idx = i
            val = val1

    return arr[idx], idx

#找出上半部分对应的两个耳线点的坐标
left_top_point = find_close(points_top_half_x, left_box_coordinate)
right_top_point = find_close(points_top_half_x, right_box_coordinate)
left_low_point = find_close(points_low_half_x, left_box_coordinate)
right_low_point = find_close(points_low_half_x, right_box_coordinate)

#得到4个耳线交点的y坐标值
left_top_point_y = points_top_half[left_top_point[1]][0]
right_top_point_y = points_top_half[right_top_point[1]][0]
left_low_point_y = points_low_half[left_low_point[1]][0]
right_low_point_y = points_low_half[right_low_point[1]][0]




#print(left_box_coordinate, right_box_coordinate)
print(left_top_point, right_top_point, left_low_point, right_low_point)
#cv2.imshow('img', img)
#cv2.imshow('result', result)
cv2.imshow('result_rotate_mask', result_rotate_mask)
#cv2.imshow('rotate_mask', rotate_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
