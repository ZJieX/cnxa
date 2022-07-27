from utils.data_manger import ImageNet100

dataset = ImageNet100(root="/data/CNX_X/data/imagenet100/imagenet100/")

with open("cls_train.txt", "w") as f:
    for idx, (path, cls_id) in enumerate(dataset.train):
        f.write(str(cls_id) + ";" + path + '\n')
    f.close()

with open("cls_test.txt", "w") as f:
    for idx, (path, cls_id) in enumerate(dataset.val):
        f.write(str(cls_id) + ";" + path + '\n')
    f.close()

with open("model_data/cls_classes.txt", "w") as f:
    for label, name in enumerate(dataset.classes_id):
        f.write(name + '\n')
    f.close()
