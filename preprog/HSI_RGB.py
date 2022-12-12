# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

# 显示汉字用
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 定义显示一张图片函数
def imshow(image):
    if image.ndim == 2:
        plt.imshow(image, cmap='gray')                     # 指定为灰度图像
    else:
        plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))


# 定义坐标数字字体及大小
def label_def():
    plt.xticks(fontproperties='Times New Roman', size=8)
    plt.yticks(fontproperties='Times New Roman', size=8)
    plt.axis('off')                                     # 关坐标，可选


# 读取图片
img_orig = cv.imread('16665.jpg', 1)    # 读取彩色图片


# RGB到HSI的变换
def rgb2hsi(image):
    b, g, r = cv.split(image)                    # 读取通道
    r = r / 255.0                                # 归一化
    g = g / 255.0
    b = b / 255.0
    eps = 1e-6                                   # 防止除零

    img_i = (r + g + b) / 3                      # I分量

    img_h = np.zeros(r.shape, dtype=np.float32)
    img_s = np.zeros(r.shape, dtype=np.float32)
    min_rgb = np.zeros(r.shape, dtype=np.float32)
    # 获取RGB中最小值
    min_rgb = np.where((r <= g) & (r <= b), r, min_rgb)
    min_rgb = np.where((g <= r) & (g <= b), g, min_rgb)
    min_rgb = np.where((b <= g) & (b <= r), b, min_rgb)
    img_s = 1 - 3*min_rgb/(r+g+b+eps)                                            # S分量

    num = ((r-g) + (r-b))/2
    den = np.sqrt((r-g)**2 + (r-b)*(g-b)+eps)
    theta = np.arccos(num/(den+eps))
    img_h = np.where((b-g) > 0, 2*np.pi - theta, theta)                           # H分量

    img_h = img_h/(2*np.pi)                                                       # 归一化
    temp_s = img_s - np.min(img_s)
    temp_i = img_i - np.min(img_i)
    img_s = temp_s/np.max(temp_s)
    img_i = temp_i/np.max(temp_i)

    image_hsi = cv.merge((img_h, img_s, img_i))
    return image_hsi


# HSI到RGB的变换
def hsi2rgb(image_hsi):
    eps = 1e-6                                                                  # 防止除零
    img_h, img_s, img_i = cv.split(image_hsi)
    # img_s = img_s / 255.0                               #归一化
    # img_i = img_i / 255.0
    # img_h = img_h / 255.0

    img_h = img_h*2*np.pi
    img_r = np.zeros(img_h.shape, dtype=np.float32)
    img_g = np.zeros(img_h.shape, dtype=np.float32)
    img_b = np.zeros(img_h.shape, dtype=np.float32)

    # 扇区1
    img_b = np.where((img_h >= 0) & (img_h < 2 * np.pi / 3), img_i * (1 - img_s), img_b)
    img_r = np.where((img_h >= 0) & (img_h < 2 * np.pi / 3),
                     img_i * (1 + img_s * np.cos(img_h) / (np.cos(np.pi/3 - img_h))), img_r)
    img_g = np.where((img_h >= 0) & (img_h < 2 * np.pi / 3), 3 * img_i - (img_r + img_b), img_g)

    # 扇区2
    img_r = np.where((img_h >= 2*np.pi/3) & (img_h < 4*np.pi/3), img_i * (1 - img_s), img_r)
    img_g = np.where((img_h >= 2*np.pi/3) & (img_h < 4*np.pi/3),
                     img_i * (1 + img_s * np.cos(img_h-2*np.pi/3) / (np.cos(np.pi/3 - (img_h-2*np.pi/3)))), img_g)
    img_b = np.where((img_h >= 2*np.pi/3) & (img_h < 4*np.pi/3), 3 * img_i - (img_r + img_g), img_b)

    # 扇区3
    img_g = np.where((img_h >= 4*np.pi/3) & (img_h <= 2*np.pi), img_i * (1 - img_s), img_g)
    img_b = np.where((img_h >= 4*np.pi/3) & (img_h <= 2*np.pi),
                     img_i * (1 + img_s * np.cos(img_h-4*np.pi/3) / (np.cos(np.pi/3 - (img_h-4*np.pi/3)))), img_b)
    img_r = np.where((img_h >= 4*np.pi/3) & (img_h <= 2*np.pi), 3 * img_i - (img_b + img_g), img_r)

    # temp_r = img_r - np.min(img_r)
    # img_r = temp_r/np.max(temp_r)
    #
    # temp_g = img_g - np.min(img_g)
    # img_g = temp_g/np.max(temp_g)
    #
    # temp_b = img_b - np.min(img_b)
    # img_b = temp_b/np.max(temp_b)

    image_out = cv.merge((img_b*255, img_g*255, img_r*255)).astype('uint8')               # 按RGB合并，后面不用转换通道
    return image_out

# def HSI2RGB(img):
#     H1, S1, I1 = img[:, :, 0] / 255.0, img[:, :, 1] / 255.0, img[:, :, 2] / 255.0
#     B = np.zeros((H1.shape[0], H1.shape[1]), dtype='float32')
#     G = np.zeros((S1.shape[0], S1.shape[1]), dtype='float32')
#     R = np.zeros((I1.shape[0], I1.shape[1]), dtype='float32')
#     H = np.zeros((H1.shape[0], H1.shape[1]), dtype='float32')
#
#     for i in range(H1.shape[0]):
#         for j in range(H1.shape[1]):
#             H = H1[i][j]
#             S = S1[i][j]
#             I = I1[i][j]
#             if (H >= 0) & (H < (np.pi * (2 / 3))):
#                 B[i][j] = I * (1 - S)
#                 R[i][j] = I * (1 + ((S * np.cos(H)) / np.cos(np.pi * (1 / 3) - H)))
#                 G[i][j] = 3 * I - (B[i][j] + R[i][j])
#
#             elif (H >= (np.pi * (2 / 3))) & (H < np.pi * (4 / 3)):
#                 R[i][j] = I * (1 - S)
#                 G[i][j] = I * (1 + ((S * np.cos(H - np.pi * (2 / 3))) / np.cos(np.pi * (1 / 2) - H)))
#                 B[i][j] = 3 * I - (G[i][j] + R[i][j])
#             elif (H >= (np.pi * (4 / 3))) & (H < (np.pi * 2)):
#                 G[i][j] = I * (1 - S)
#                 B[i][j] = I * (1 + ((S * np.cos(H - np.pi * (4 / 3))) / np.cos(np.pi * (10 / 9) - H)))
#                 R[i][j] = 3 * I - (G[i][j] + B[i][j])
#     img = cv.merge((B * 255, G * 255, R * 255))
#     img = img.astype('uint8')
#     return img



if __name__ == '__main__':                                       # 运行当前函数
    hsi = rgb2hsi(img_orig)                             # RGB到HSI的变换
    h,s,i = cv.split(hsi)
    img_revise = hsi2rgb(hsi)                        # HSI复原到RGB

    # h, s, i = cv.split(cv.cvtColor(img_orig, cv.COLOR_BGR2HSV))       # 自带库函数HSV模型
    im_b, im_g, im_r = cv.split(img_orig)                        # 获取RGB通道数据

    plt.subplot(221), plt.imshow(cv.cvtColor(img_orig, cv.COLOR_BGR2RGB)), plt.title('originimg'), label_def()
    plt.subplot(222), plt.imshow(im_r, 'gray'), plt.title('R'), label_def()
    plt.subplot(223), plt.imshow(im_g, 'gray'), plt.title('G'), label_def()
    plt.subplot(224), plt.imshow(im_b, 'gray'), plt.title('B'), label_def()
    plt.show()

    plt.subplot(221), plt.imshow(hsi), plt.title('HSI'), label_def()
    plt.subplot(222), plt.imshow(h, 'gray'), plt.title('H'), label_def()
    plt.subplot(223), plt.imshow(s, 'gray'), plt.title('S'), label_def()
    plt.subplot(224), plt.imshow(i, 'gray'), plt.title('I'), label_def()
    plt.show()

    plt.subplot(121), plt.imshow(cv.cvtColor(img_orig, cv.COLOR_BGR2RGB)), plt.title('originRGB'), label_def()
    plt.subplot(122), plt.imshow(cv.cvtColor(img_revise, cv.COLOR_BGR2RGB)), plt.title('HSItoRGB'), label_def()
    plt.show()

