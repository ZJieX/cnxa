# 根据自己得数据集形式来
import os
import os.path as osp
import numpy as np
import random
import xml.etree.ElementTree as ET


class ImageNet100(object):
    def __init__(self, root='data', **kwargs):
        self.dataset_dir = root

        self._check_before_run()

        train, val, total, num_classes, classes_id = self._process_dir()

        train_labels = self._set_label(train)  # 训练集中的种类数
        val_labels = self._set_label(val)  # 验证集中的种类数
        self._logs(len(train), len(val), train_labels, val_labels, total, num_classes)

        self.train = train
        self.val = val
        self.total = total
        self.num_classes = num_classes
        self.classes_id = classes_id
        # self.train_label = train_label
        # self.val_label = val_label

    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))

    def _process_dir(self):
        dic = {}
        TDL = []
        VDL = []

        for i, Fname in enumerate(os.listdir(self.dataset_dir)):
            dic[Fname] = i
            path = os.path.join(self.dataset_dir, Fname)

            for fname in os.listdir(path):
                absolute_path = path + '/' + fname
                information = absolute_path
                TDL.append((information, dic[Fname]))
        random.shuffle(TDL)
        total = len(TDL)  # 总数据量

        classes = len(dic)  # 总类别数
        """
        with open('train.txt', "w") as f:
            for lins in DL:
                f.write(lins + '\n')
            f.close()
        """

        test_num = round(0.1 * total)  # 验证集比例占总数据0.1

        for i in range(test_num):
            num_train = len(TDL)  # 更新总数据长度
            index = random.randint(0, num_train - 1)  # 从现有长度中随机取一个数据作为验证集

            VDL.append(TDL[index])

            TDL.pop(index)  # 删除已经分配到val里面的数据

        return TDL, VDL, total, classes, dic

    def _logs(self, train_num, val_num, train_label, val_label, total, num_classes):
        print("==> ImageNet100 loaded")
        print("Dateset statistics")

        print("   -------------------------------")
        print("   subset  | # labels | # images")
        print("   -------------------------------")
        print("   total   |  {:5d} \t|{:8d}".format(num_classes, total))
        print("   train   |  {:5d} \t|{:8d}".format(train_label, train_num))
        print("   val     |  {:5d} \t|{:8d}".format(val_label, val_num))
        print("   -------------------------------")

    def _set_label(self, x):
        save = []
        for i, (p, l) in enumerate(x):
            save.append(l)

        return len(set(save))


class VOC2021(object):
    """
    使用VOC数据集进行多标签分类时，必须要传入图片所在的上一级的绝对路径
    """
    def __init__(self, anno='xml', img_path='img', classes='voc2021'):
        super(VOC2021, self).__init__()
        self.annotations_dir = anno
        self.image_dir = img_path
        self.voc2021 = classes

        self.dict_ = dict()

        # 映射类别
        with open(self.voc2021, encoding='utf-8') as f:
            class_names = f.readlines()

        self.class_names = [c.strip() for c in class_names]
        self.classes_num = len(self.class_names)

        for i, name in enumerate(self.class_names):
            self.dict_[name] = i

        self.train, self.val, self.total = self._process()
        self._logs(len(self.train), len(self.val), len(self.dict_), len(self.dict_), self.total, len(self.dict_))

    def _getVOC2021Info(self, xmlFile, img_dir):
        N = list()
        # print("Extracting the category for {}".format(xmlFile.split('/')[-1].split('.')[0]))  # 对xml文件名称进行提取
        root = ET.parse(xmlFile).getroot()

        anns = root.findall('object')
        img_name = root.find('filename').text
        image_path = osp.join(img_dir, img_name)

        for ann in anns:
            name = ann.find('name').text
            N.append(name)

        return image_path, N

    def _process(self):
        TDL = list()
        VDL = list()

        for _, _, fname in os.walk(self.annotations_dir):
            for i in range(len(fname)):

                img_path, labels = self._getVOC2021Info(os.path.join(self.annotations_dir, fname[i]), self.image_dir)
                # print("{}, {}".format(img_path, labels))
                l_list = [0 for x in range(self.classes_num)]
                for l in labels:
                    if l == 'w':
                        continue
                    l_list[self.dict_[l]] = 1

                TDL.append((img_path, l_list))

        random.shuffle(TDL)
        test_num = round(0.1 * len(TDL))  # 验证集比例占总数据0.1
        total = len(TDL)
        for i in range(test_num):
            num_train = len(TDL)  # 更新总数据长度
            index = random.randint(0, num_train - 1)  # 从现有长度中随机取一个数据作为验证集

            VDL.append(TDL[index])

            TDL.pop(index)  # 删除已经分配到val里面的数据

        return TDL, VDL, total

    def _logs(self, train_num, val_num, train_label, val_label, total, num_classes):
        print("==> VOC2021 loaded")
        print("Dateset statistics")

        print("   -------------------------------")
        print("   subset  | # labels | # images")
        print("   -------------------------------")
        print("   total   |  {:5d} \t|{:8d}".format(num_classes, total))
        print("   train   |  {:5d} \t|{:8d}".format(train_label, train_num))
        print("   val     |  {:5d} \t|{:8d}".format(val_label, val_num))
        print("   -------------------------------")


class COCO(object):
    def __init__(self):
        super(COCO, self).__init__()
        pass


"""
if __name__ == '__main__':
    data = VOC2021(anno='D://PyCharm/xiaozhijie/CNXA/CNX_V3/dataset/VOCdevkit/VOC2021/Annotations/',
                   img_path='D://PyCharm/xiaozhijie/CNXA/CNX_V3/dataset/VOCdevkit/VOC2021/JPEGImages/',
                   classes='D://PyCharm/xiaozhijie/CNXA/CNX_V3/model_data/voc2021_classes.txt')
    # print("{}, {}".format(data.train, data.val))
    # print("{}, {}, {}".format(len(data.train), len(data.val), data.total))
"""