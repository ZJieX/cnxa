from utils.data_manger import ImageNet100, VOC2021, COCO2017

# ImageNet100 = ImageNet100(root="E://self_dataset/imagenet100/imagenet100/")
# VOC2021 = VOC2021(anno='D://PyCharm/xiaozhijie/CNXA_VOC2021/dataset/dataset/VOCdevkit/VOC2021/Annotations/',
#                           img_path='D://PyCharm/xiaozhijie/CNXA_VOC2021/dataset/VOCdevkit/VOC2021/JPEGImages/',
#                           classes='D://PyCharm/xiaozhijie/CNXA_VOC2021/model_data/coco2017_classes.txt')
COCO2017 = COCO2017(train_dir='/data/CNXA_COCO/dataset/COCO2VOC/train/Annotations/',
                   val_dir='/data/CNXA_COCO/dataset/COCO2VOC/val/Annotations/',
                   train_img='/data/CNXA_COCO/dataset/COCO2VOC/train/JPEGImages/',
                   val_img='/data/CNXA_COCO/dataset/COCO2VOC/val/JPEGImages/')


def creat_txt(dataset, name='data'):
    with open("cls_train_" + name + ".txt", "w") as f:
        for idx, (path, cls_id) in enumerate(dataset.train):
            f.write(str(cls_id) + ";" + path + '\n')
        f.close()

    with open("cls_test_" + name + ".txt", "w") as f:
        for idx, (path, cls_id) in enumerate(dataset.val):
            f.write(str(cls_id) + ";" + path + '\n')
        f.close()

    with open("model_data/coco2017_classes.txt", "w") as f:
        for label, name in enumerate(dataset.classes_id):
            f.write(name + '\n')
        f.close()


if __name__ == '__main__':
    creat_txt(COCO2017, name='coco2017')
