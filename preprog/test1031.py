#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 在HSI域对强度值进行模糊化处理

import cv2 as cv
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from preprog.HSI_RGB import rgb2hsi,hsi2rgb



def t1fs(max,min,mean,num):
    # return  num
    if num <= mean:
        num = (num-min)/(mean-min)
    elif num >= mean:
        num = (max-num)/(max-mean)
    return num


def t1transI(image):
    hsi = rgb2hsi(image)
    h,s,i = cv.split(hsi)
    # img_i = np.trunc(np.asarray(i)*255)
    img_i = np.asarray(i)
    max_i = img_i.max()
    min_i = img_i.min()
    mean_i = img_i.mean()
    # print("max:{},min:{},mean:{}".format(max_i,min_i,mean_i))
    function_vector = np.vectorize(t1fs)
    new_i = function_vector(max_i,min_i,mean_i,img_i)
    # new_i = np.sqrt(temp_i)
    # print("new_i{}".format(new_i))
    image_hsi = cv.merge([h, s, new_i])
    image_out = hsi2rgb(image_hsi)
    # image_out = Image.fromarray(np.uint8(image_out*255))
    return image_out

if __name__ == '__main__':
    img = cv.imread('./16665.jpg')
    # plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    # plt.show()
    newimg = t1transI(img)
    plt.imshow(cv.cvtColor(newimg, cv.COLOR_BGR2RGB))
    plt.show()
