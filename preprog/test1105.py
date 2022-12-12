#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 在HSI域对强度值进行模糊化处理
import time

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from numpy import exp

from preprog.HSI_RGB import rgb2hsi,hsi2rgb


def t1fs(max,min,mean,num):
    f=1-mean
    if num <= f:
        u = ((num-min)/(f-min))**0.5
        nums = u*(f-min)+min
    elif num > f:
        u = ((max-num)/(max-f))**0.5
        nums = max - u*(max-f)
    return nums

def t2fs(var,mean,median,num):
# 通过中位数、均值确定上下隶属度位置
# 高斯函数
    if median <= mean:
        gauss_up = exp(-(((mean - num) ** 2) / (2 * var ** 2)))
        gauss_down = (median / mean) * exp(-(((mean - num) ** 2) / (2 * var ** 2)))
    else:
        gauss_up = exp(-(((mean - num) ** 2) / (2 * var ** 2)))
        gauss_down = (mean / median) * exp(-(((mean - num) ** 2) / (2 * var ** 2)))
    return (gauss_up + gauss_down) / 2

def t1transI(image):
    hsi = rgb2hsi(image)
    h,s,i = cv.split(hsi)
    img_i = np.asarray(i)
    max_i = img_i.max()
    min_i = img_i.min()
    mean_i = img_i.mean()
    function_vector = np.vectorize(t1fs)
    new_i = function_vector(max_i,min_i,mean_i,img_i)
    image_hsi = cv.merge([h, s, new_i])
    image_out = hsi2rgb(image_hsi)
    # image_out = Image.fromarray(np.uint8(image_out*255))
    return image_out

def t2ftransI(image):
    hsi = rgb2hsi(image)
    h,s,i = cv.split(hsi)
    img_i = np.asarray(i)
    mean_i = img_i.mean()
    var_i = img_i.var()
    x = img_i.reshape( 1, -1)
    median_i = np.median(x)
    function_vector = np.vectorize(t2fs)
    new_i = function_vector(var_i,mean_i,median_i,img_i)
    image_hsi = cv.merge([h, s, new_i])
    image_out = hsi2rgb(image_hsi)
    # image_out = Image.fromarray(np.uint8(image_out*255))
    return image_out

def histogram_demo(image):
    plt.hist(image.ravel(), 256, [0, 256])#ravel函数功能是将多维数组降为一维数组
    plt.show()
def image_hist(image): #画三通道图像的直方图
   color = ("blue", "green", "red")#画笔颜色的值可以为大写或小写或只写首字母或大小写混合
   for i, color in enumerate(color):
       hist = cv.calcHist([image], [i], None, [256], [0, 256])
       plt.plot(hist, color=color)
       plt.xlim([0, 256])
   plt.show()


if __name__ == '__main__':
    img = cv.imread('./16665.jpg')
    start_time = time.perf_counter()
    newimg = t2ftransI(img)
    end_time = time.perf_counter()
    print(end_time - start_time)
    histogram_demo(img)
    histogram_demo(newimg)
    plt.subplot(211),plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(212),plt.imshow(cv.cvtColor(newimg, cv.COLOR_BGR2RGB))
    plt.show()

    # y = t1fs(1.0,0.,0.5,0.7)
    # print(y)