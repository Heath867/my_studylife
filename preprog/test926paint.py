#!/usr/bin/env python
# -*- coding: utf-8 -*-


import cv2 as cv
import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy import linspace

from preprog.pyit2fls import IT2FS_Semi_Elliptic, IT2FS_plot, IT2FS_Gaussian_UncertStd, L_IT2FS_Gaussian_UncertStd, \
    R_IT2FS_Gaussian_UncertStd


def main():    # 读取图像并判断是否读取成功
    domain = linspace(0., 1., 100)
    mySet = IT2FS_Gaussian_UncertStd(domain, [0.5, 0.2, 0.05, 1.])
    mySet.plot()

if __name__ == '__main__':
    main()