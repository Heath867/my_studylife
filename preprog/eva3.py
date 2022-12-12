import sys
import  matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from numpy import linspace, maximum, minimum

from pyit2fls import IT2FS, trapezoid_mf


def trapezoid_mf(x, params):
    if (params[1] - params[0]!=0)*(params[3] - params[2]!=0)==1:
        return minimum(1, maximum(0, ((((params[4] * ((x - params[0]) / (params[1] - params[0]))) * (x <= params[1])) +
                                       ((params[4] * ((params[3] - x) / (params[3] - params[2]))) * (x >= params[2]))) +
                                      (params[4] * ((x > params[1]) * (x < params[2]))))))
    else: return x/255

def evaluate_up(x, upmftype, params1, downmftype, params2):
    if upmftype == trapezoid_mf:
        x1 = trapezoid_mf(x, params1)
        # print(x1)
    else: return False
    if downmftype == trapezoid_mf:
        x2 = trapezoid_mf(x, params2)
        # print(x2)
    else: return False
    return (x1 + x2)/2

def eval(A,x):
    # A = [50,120,60,155,210,220,90,190]
    B = sorted(A)
    p1 = []
    p2 = []
    for i in range(4) :
        if i %2 == 0 :
            p1.append(B[i])
            p1.append(B[7-i])
        else:
            p2.append(B[i])
            p2.append(B[7-i])

    # print(B)
    print("p1:{}".format(p1),"p2:{}".format(p2))
    p1 = sorted(p1)
    p2 = sorted(p2)
    p1.append(1.)
    p2.append(0.9)
    # print(p1)
    # print(p2)
    # myIT2FS = IT2FS(linspace(0., 255., 256), trapezoid_mf, p1, trapezoid_mf, p2)
    # myIT2FS.plot()
    return 255*evaluate_up(x, trapezoid_mf, p1 ,trapezoid_mf, p2)


def main():
    # A = [50, 120, 60, 155, 210, 220, 90, 190]
    # print(eval(A,177))
    img = cv.imread('./5547758_eea9edfd54_n.jpg')
    if img is None:
        print('Failed to read file')
        sys.exit()

    else:
        # print(img)
        plt.imshow(img)
        plt.show()
        x=img.shape[:2]
        top_size, bottom_size, left_size, right_size = (1, 1, 1, 1)
        img = cv.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv.BORDER_REPLICATE)
        zeros = np.ones(img.shape[:2], dtype=img.dtype)
        for w in range(3):
            # cv.imshow('zeros',zeros)
            for i in range(1, img.shape[0]-2):
                for j in range(1, img.shape[1]-2):

                    A = [img[i - 1, j, w], img[i + 1, j, w], img[i, j - 1, w], img[i, j + 1, w],
                         img[i - 1, j - 1, w], img[i - 1, j + 1, w], img[i + 1, j - 1, w], img[i + 1, j + 1, w] ]
                    # print(A)

                    zeros[i, j] = eval(A,img[i,j,w])
                    # print(zeros)
                    # result_BGR_alpha = cv.merge([img, zeros])
            plt.imshow(zeros)
            plt.show()

    # print('原图的通道数为：{}'.format(img.shape[2]))
    # print('处理后的通道数为：{}'.format(result_BGR_alpha.shape[2]))
    # # 图像保存到硬盘
    # cv.imwrite('./5547758_eea9edfd54_n_f.jpg', result_BGR_alpha)

if __name__ == '__main__':
    main()