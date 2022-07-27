import numpy as np
import torch
import time
import datetime
from PIL import Image

from classification import (Classification, cvtColor,
                            letterbox_image, preprocess_input)
from utils.utils import letterbox_image


class top1_Classification(Classification):
    def detect_image(self, image):
        image = cvtColor(image)  # 将读取的图片转化为RGB

        # 以下操作是对图片进行不失真resize
        image_data = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])

        # 归一化+添加上batch_size维度+转置
        image_data = np.transpose(np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0), (0, 3, 1, 2))

        with torch.no_grad():
            photo = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.cuda:
                photo = photo.cuda()
            # 图片传入网络进行预测
            preds = torch.softmax(self.model(photo)[0], dim=-1).cpu().numpy()

        arg_pred = np.argmax(preds)
        return arg_pred


class top5_Classification(Classification):
    def detect_image(self, image):
        image = cvtColor(image)  # 将读取的图片转化为RGB

        # 以下操作是对图片进行不失真resize
        image_data = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])

        # 归一化+添加上batch_size维度+转置
        image_data = np.transpose(np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0), (0, 3, 1, 2))

        with torch.no_grad():
            photo = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.cuda:
                photo = photo.cuda()

            # 图片传入网络进行预测
            preds = torch.softmax(self.model(photo)[0], dim=-1).cpu().numpy()

        arg_pred = np.argsort(preds)[::-1]
        arg_pred_top5 = arg_pred[:5]
        return arg_pred_top5


def evaluteTop1(classfication, lines):
    correct = 0
    total = len(lines)
    for index, line in enumerate(lines):
        annotation_path = line.split(';')[1].split()[0]
        x = Image.open(annotation_path)
        y = int(line.split(';')[0])

        pred = classfication.detect_image(x)
        correct += pred == y
        if index % 100 == 0:
            print("[%d/%d]" % (index, total))
    return correct / total


def evaluteTop5(classfication, lines):
    correct = 0
    total = len(lines)
    for index, line in enumerate(lines):
        annotation_path = line.split(';')[1].split()[0]
        x = Image.open(annotation_path)
        y = int(line.split(';')[0])

        pred = classfication.detect_image(x)
        correct += y in pred
        if index % 100 == 0:
            print("[%d/%d]" % (index, total))
    return correct / total


classfication_top1 = top1_Classification()
classfication_top5 = top5_Classification()
with open("./cls_test.txt", "r") as f:
    lines = f.readlines()

start_time = time.time()
backbone = Classification.get_defaults("backbone")
aa = Classification.get_defaults("aa")
model = Classification.get_defaults("model_path")

top1 = evaluteTop1(classfication_top1, lines)
print("top-1 accuracy = %.2f%%" % (top1 * 100))
top5 = evaluteTop5(classfication_top5, lines)

elapsed = round(time.time() - start_time)
elapsed = str(datetime.timedelta(seconds=elapsed))

with open("evaluate_result/{}{}.txt".format(backbone, aa), "a") as f:
    f.write("{} is being used for evaluation"
            "evaluate {}{} with spend time:{} =====> top1 acc:{}\ttop5 acc:{}\n".format(model, backbone, aa, elapsed, top1, top5))
    f.close()
print("top-1 accuracy = %.2f%%\ttop-5 accuracy = %.2f%%" % (top1 * 100, top5 * 100))

