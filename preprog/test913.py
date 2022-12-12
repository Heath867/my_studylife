#!/usr/bin/env python
# -*- coding: utf-8 -*-


import cv2 as cv
import sys
import numpy as np
import matplotlib.pyplot as plt


def main():    # 读取图像并判断是否读取成功

    img = cv.imread('./5547758_eea9edfd54_n.jpg')
    x=img.shape[:2]

    print('old pic:{}'.format(img))
    top_size, bottom_size, left_size, right_size = (x[1], x[1], x[0], x[0])
    img = cv.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv.BORDER_REPLICATE)
    print('new:{}'.format(img))
    plt.imshow(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for w in range(img.shape[2]):
                A = [img[i-1,j,w],img[i+1,j,w],img[i,j-1,w],img[i,j+1,w],
                     img[i-1,j-1,w],img[i-1,j+1,w],img[i+1,j-1,w],img[i+1,j+1,w],]
                zeros = np.ones(img.shape[:2], dtype=img.dtype)
                zeros[i,j] =

if __name__ == '__main__':
    main()