from utils.data_manger import ImageNet100, VOC2021

# ImageNet100 = ImageNet100(root="E://self_dataset/imagenet100/imagenet100")
VOC2021 = VOC2021(anno='/data/CNXA_VOC2021/dataset/VOCdevkit/VOC2021/Annotations/',
                          img_path='/data/CNXA_VOC2021/dataset/VOCdevkit/VOC2021/JPEGImages/',
                          classes='/data/CNXA_VOC2021/model_data/voc2021_classes.txt')


def creat_txt(dataset, name='data'):
    with open("cls_train_" + name + ".txt", "w") as f:
        for idx, (path, cls_id) in enumerate(dataset.train):
            f.write(str(cls_id) + ";" + path + '\n')
        f.close()

    with open("cls_test_" + name + ".txt", "w") as f:
        for idx, (path, cls_id) in enumerate(dataset.val):
            f.write(str(cls_id) + ";" + path + '\n')
        f.close()

    if str(dataset) == "ImageNet100" :
        with open("model_data/cls_classes.txt", "w") as f:
            for label, name in enumerate(dataset.classes_id):
                f.write(name + '\n')
            f.close()


if __name__ == '__main__':
    creat_txt(VOC2021, name='voc2021')
