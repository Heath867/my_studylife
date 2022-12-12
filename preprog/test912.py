#!/usr/bin/env python
# -*- coding: utf-8 -*-


import cv2 as cv
import sys
import numpy as np
import matplotlib.pyplot as plt


def main():    # 读取图像并判断是否读取成功

    img = cv.imread('./5547758_eea9edfd54_n.jpg')
    if img is None:
        print('Failed to read file')
        sys.exit()
    else:
        # 使用cv.merge()函数添加alpha通道
        zeros = np.ones(img.shape[:2], dtype=img.dtype) * 100
        result_BGR_alpha = cv.merge([img, zeros])
        print('原图的通道数为：{}'.format(img.shape[2]))
        print('处理后的通道数为：{}'.format(result_BGR_alpha.shape[2]))

        # 图像保存到硬盘
        cv.imwrite('./5547758_eea9edfd54_n_fuzzy.jpg', result_BGR_alpha)

        # 以下代码为图像展示
        # 因为opencv的颜色通道顺序为[B,G,R]，而matplotlib的颜色通道顺序为[R,G,B]，
        # 所以作图前要先进行通道顺序的调整。
        result_RGB_alpha = result_BGR_alpha[:, :, (2, 1, 0, 3)]
        img_RGB = img[:, :, (2, 1, 0)]

        # 以下开始绘制图形并显示
        plt.figure()
        plt.subplot(1, 2, 1)  # 将画板分为一行两列，接下来要绘的图位于第一个位置
        plt.title('Original image')
        plt.imshow(img_RGB)
        plt.subplot(1, 2, 2)  # 将画板分为一行两列，接下来要绘的图位于第二个位置
        plt.title('Add alpha channel image')
        plt.imshow(result_RGB_alpha)
        plt.show()

if __name__ == '__main__':
    main()