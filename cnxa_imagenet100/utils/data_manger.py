# 根据自己得数据集形式来
import os
import os.path as osp
import numpy as np
import random


class ImageNet100(object):
    def __init__(self, root='data', **kwargs):
        self.dataset_dir = root

        self._check_before_run()

        train, val, total, num_classes, classes_id = self._process_dir()

        train_labels = self._set_label(train)   # 训练集中的种类数
        val_labels = self._set_label(val)   # 验证集中的种类数
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


"""
if __name__ == '__main__':
    data = ImageNet100(root='E://self_dataset/imagenet100/imagenet100/')
    print(len(data.val))
    # print("{}, {}, {}".format(len(data.train), len(data.val), data.total))
"""