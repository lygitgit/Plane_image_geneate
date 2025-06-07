import cv2
import numpy as np

# 初始化全局变量
drawing = False  # 鼠标是否按下
ix, iy = -1, -1  # 初始坐标
lines = []  # 存储线的起点和终点坐标
img_w = 0
img_h = 0

dir = "processed_data/background_2"
img = cv2.imread(dir + '/background.png')
# 获取图像的尺寸
height, width, _ = img.shape
block_size = 800

# 鼠标回调函数
def draw_line(event, x, y, flags, param):
    global ix, iy, drawing, lines

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # 在图片上绘制临时线条
            temp_img = img.copy()
            cv2.line(temp_img, (ix, iy), (x, y), (0, 0, 255), 2)
            cv2.imshow('image', temp_img)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # 记录线条的起点和终点
        lines.append(((ix + img_w * block_size, iy + img_h * block_size), (x + img_w * block_size, y + img_h * block_size)))
        print(((ix + img_w * block_size, iy + img_h * block_size), (x + img_w * block_size, y + img_h * block_size)))
        cv2.imshow('image', img)

# 读取已有的图片

for h in range(0, height, block_size):
    for w in range(0, width, block_size):
        # 计算当前块的右下角坐标
        h_star = max(h, 0)
        w_star = max(w, 0)
        h_end = min(h + block_size, height)
        w_end = min(w + block_size, width)

        # 提取当前块
        block = img[h_star:h_end, w_star:w_end]
        # 创建一个窗口并设置鼠标回调函数
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', draw_line)
        while True:
            cv2.imshow('image', block)
            if cv2.waitKey(1) & 0xFF == 27:  # 按下ESC键退出
                break
        cv2.destroyAllWindows()
        img_w += 1
    img_h += 1
    img_w = 0


import math
output_file_path = dir + '/garage_deposit_location.txt'
with open(output_file_path, 'w') as file:
    for line in lines:
        rad = math.atan2(line[1][1] - line[0][1], line[1][0] - line[0][0])
        angle = np.rad2deg(rad)
        angle = -angle - 90
        if angle <= -180:
            angle += 360
        file.write(f"{line[0][0], line[0][1], angle}\n")
