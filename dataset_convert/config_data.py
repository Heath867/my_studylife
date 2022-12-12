dataset_type = 'COCODataset'
classes = {'mandatory', 'prohibitory', 'warning'}
data_root = '/storage/YYT/mmdetection-master'
data = dict(
    train=dict(
        type='CocoDataset',
        ann_file='/storage/YYT/mmdetection-master/CCTSDB_2021/annotations.json',
        classes=classes,
        img_prefix='/storage/YYT/mmdetection-master/CCTSDB_2021/images',
    ),
    val = dict(
        type='CocoDataset',
        ann_file='/storage/YYT/mmdetection-master/CCTSDB_2021/annotations.json',
        classes=classes,
        img_prefix='/storage/YYT/mmdetection-master/CCTSDB_2021/images',
    ),
    test=dict(
        type='CocoDataset',
        ann_file='/storage/YYT/mmdetection-master/CCTSDB_2021/annotations.json',
        classes=classes,
        img_prefix='/storage/YYT/mmdetection-master/CCTSDB_2021/images',
    )
)