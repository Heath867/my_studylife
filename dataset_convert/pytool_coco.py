import os
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
#这个有用-，-
#最终版查看标注信息
coco_coco=COCO('/storage/YYT/mmdetection-master/data/coco/annotations/instances_test2017.json')
catNms = ['mandatory', 'prohibitory']
CatIds = coco_coco.getCatIds(catNms=catNms)
imgIds = coco_coco.getImgIds(catIds=CatIds)

for j in range(len(imgIds)):
    img = coco_coco.loadImgs(imgIds)[j]
    img_path = os.path.join('/storage/YYT/mmdetection-master/data/coco/train2017',img['file_name'])
    I = cv2.imread(img_path)
    I= I[:, :, ::-1]
    annIds = coco_coco.getAnnIds(imgIds=img['id'],catIds=CatIds,iscrowd=None)
    anns = coco_coco.loadAnns(annIds)
    for i in range(len(annIds)):
        x, y, w, h = anns[i]['bbox']  # 读取边框
        p1 = [int(x), int(y)]
        p2 = [int(x + w), int(y + h)]
        I = cv2.rectangle(I.copy(), p1, p2, (0, 255, 255), 2)
    plt.imshow(I.copy())
    plt.show()