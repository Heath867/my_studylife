import os.path
import json
import xml.dom.minidom
import cv2

data_dir = '../CCTSDB_2021'
image_file_dir = os.path.join(data_dir,'images')
xml_file_dir = os.path.join(data_dir,'xml/xml')

image_list = [image_file_name.split('.')[0]
              for image_file_name in os.listdir(image_file_dir)]
# 初始化id
image_id = 1
annotation_id = 1
# 初始化输出
coco_output = {
    "images":[],
    "type": "instances",
    "categories": [],
    "annotations": [],
}

categories_map = {'mandatory': 1, 'prohibitory': 2, 'warning': 3}

for key in categories_map:
    category_info = {"id": categories_map[key], "name": key}
    coco_output['categories'].append(category_info)

# 遍历文件
for i,file_name in enumerate(image_list):
    image_file_name = file_name + '.jpg'
    xml_file_name = file_name + '.xml'
    image_file_path = os.path.join(image_file_dir, image_file_name)
    xml_file_path = os.path.join(xml_file_dir, xml_file_name)
    image = cv2.cvtColor(cv2.imread(image_file_path), cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape

    # 获取图片信息 file_name,height,width
    image_dict = {
        'file_name': file_name,
        'height': height,
        'width': width,
        'id': image_id,
    }
    coco_output["images"].append(image_dict)

    DOMTree = xml.dom.minidom.parse(xml_file_path)
    collection = DOMTree.documentElement

    names = collection.getElementsByTagName('name')
    names = [name.firstChild.data for name in names]

    xmins = collection.getElementsByTagName('xmin')
    xmins = [xmin.firstChild.data for xmin in xmins]
    ymins = collection.getElementsByTagName('ymin')
    ymins = [ymin.firstChild.data for ymin in ymins]
    xmaxs = collection.getElementsByTagName('xmax')
    xmaxs = [xmax.firstChild.data for xmax in xmaxs]
    ymaxs = collection.getElementsByTagName('ymax')
    ymaxs = [ymax.firstChild.data for ymax in ymaxs]

    object_num = len(names)
    ### 获取图片对应boxs信息box
    for j in range(object_num):
        if names[j] in categories_map:
            if names[0] == '?':
                x1, y1, x2, y2 = int(float(xmins[j-1])), int(float(ymins[j-1])), int(float(xmaxs[j-1])), int(float(ymaxs[j-1]))
                x1, y1, x2, y2 = x1 - 1, y1 - 1, x2 - 1, y2 - 1

                if x2 == width:
                    x2 -= 1
                if y2 == height:
                    y2 -= 1

                x, y = x1, y1
                w, h = x2 - x1 + 1, y2 - y1 + 1
                category_id = categories_map[names[j]]
                area = w * h
                ann_dict = {
                    "id":annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,# 类别id
                    "bbox": [x,y,w,h],#[x,y,w,h]
                    "area": area,
                    "iscrowd": 0,#单人或者多人
                }
            else:
                x1, y1, x2, y2 = int(float(xmins[j])), int(float(ymins[j])), int(float(xmaxs[j])), int(float(ymaxs[j]))
                x1, y1, x2, y2 = x1 - 1, y1 - 1, x2 - 1, y2 - 1

                if x2 == width:
                    x2 -= 1
                if y2 == height:
                    y2 -= 1

                x, y = x1, y1
                w, h = x2 - x1 + 1, y2 - y1 + 1
                category_id = categories_map[names[j]]
                area = w * h
                ann_dict = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,  # 类别id
                    "bbox": [x, y, w, h],  # [x,y,w,h]
                    "area": area,
                    "iscrowd": 0,  # 单人或者多人
                }
            coco_output["annotations"].append(ann_dict)
            annotation_id += 1

    image_id += 1
# 保存
with  open('./annotations.json', 'w')  as f:
    json.dump(coco_output, f, indent=4)

print('---整理后的标注文件---')
print('所有图片的数量：',  len(coco_output['images']))
print('所有标注的数量：',  len(coco_output['annotations']))
print('所有类别的数量：',  len(coco_output['categories']))