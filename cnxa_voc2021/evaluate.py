import numpy as np
import torch
import time
import datetime
from PIL import Image

from classification import (Classification, cvtColor,
                            letterbox_image, preprocess_input)
from utils.utils import letterbox_image, compute_mAP
from sklearn.metrics import average_precision_score


class classification(Classification):
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
            # preds = torch.softmax(self.model(photo)[0], dim=-1).cpu().numpy()
            preds = self.model(photo)

        return preds


def evaluate_mAP(classfication, lines):
    print("=======> Start calculating mAP")
    total = len(lines)
    AP = list()
    for index, line in enumerate(lines):
        annotation_path = line.split(';')[1].split()[0]
        x = Image.open(annotation_path)
        y = list(line.split(';')[0])[1: len(line.split(';')[0]) - 1]
        y = y[0: len(y): 3]
        labels = list(map(int, y))
        labels = np.array(labels)
        labels = torch.from_numpy(labels)

        pred = classfication.detect_image(x)

        y_true = labels.cpu().numpy()
        y_pred = pred.cpu().numpy()

        AP.append(average_precision_score(y_true, y_pred[0]))
        if index % 100 == 0:
            print("[%d/%d]" % (index, total))
    return np.mean(AP)


def evaluate_mAcc(classfication, lines):
    print("=======> Start calculating macc")
    total = len(lines)
    acc = list()
    for index, line in enumerate(lines):
        annotation_path = line.split(';')[1].split()[0]
        x = Image.open(annotation_path)
        y = list(line.split(';')[0])[1: len(line.split(';')[0]) - 1]
        y = y[0: len(y): 3]
        labels = list(map(int, y))
        labels = np.array(labels)
        labels = torch.from_numpy(labels)

        pred = classfication.detect_image(x)
        pred = (pred > 0)
        labels = (labels == 1)
        # print(pred[0])
        # print(labels)

        acc.append((np.sum(pred[0].cpu().numpy() == labels.cpu().numpy()) / len(labels)).astype(float))
        if index % 100 == 0:
            print("[%d/%d]" % (index, total))
    macc = sum(acc) / len(acc)
    return macc
    

def evaluate_f1(classfication, lines):
    print("=======> Start calculating f1")
    f1 = list()
    precision = list()
    recall = list()
    total = len(lines)
    a = 1e-9

    for index, line in enumerate(lines):
        annotation_path = line.split(';')[1].split()[0]
        x = Image.open(annotation_path)
        y = list(line.split(';')[0])[1: len(line.split(';')[0]) - 1]
        y = y[0: len(y): 3]
        labels = list(map(int, y))
        labels = np.array(labels)
        labels = torch.from_numpy(labels)

        pred = classfication.detect_image(x)
        pred = (pred > 0)

        TP = np.sum((pred[0].cpu().numpy() == True) & (labels.cpu().numpy() == 1))
        # print(TP)
        FN = np.sum((pred[0].cpu().numpy() == False) & (labels.cpu().numpy() == 1))
        # print(FN)
        FP = np.sum((pred[0].cpu().numpy() == True) & (labels.cpu().numpy() == 0))
        # print(FP)

        precision.append(TP / (TP + FP + a))
        recall.append(TP / (TP + FN + a))
        f1.append((2 * precision[-1] * recall[-1]) / (precision[-1] + recall[-1] + a))
        if index % 100 == 0:
            print("[%d/%d]" % (index, total))

    mf1 = sum(f1) / len(f1)
    mprecision = sum(precision) / len(precision)
    mrecall = sum(recall) / len(recall)
 
    return mf1, mprecision, mrecall


classfication = classification()
with open("./cls_test_voc2021.txt", "r") as f:
    lines = f.readlines()

start_time = time.time()
backbone = classification.get_defaults("backbone")
aa = classification.get_defaults("aa")
model = classification.get_defaults("model_path")

mAP = evaluate_mAP(classfication, lines)
mAcc = evaluate_mAcc(classfication, lines)
f1, precision, recall = evaluate_f1(classfication, lines)

elapsed = round(time.time() - start_time)
elapsed = str(datetime.timedelta(seconds=elapsed))

with open("evaluate_result/{}{}.txt".format(backbone, aa), "a") as f:
    f.write("\n{} is being used for evaluation"
            "evaluate {}{} with spend time:{} =====> mAP:{}"
            "\tmAcc: {}"
            "\tPrecision: {}"
            "\tRecall: {}"
            "\tf1: {}".format(model, backbone, aa, elapsed, mAP * 100, mAcc * 100, precision * 100, recall * 100, f1 * 100))
    f.close()
print("mAP = %.2f%%" % (mAP * 100))
print("mAcc = %.2f%%" % (mAcc * 100))
print("f1 = %.2f%%" % (f1 * 100))
print("Precision = %.2f%%" % (precision * 100))
print("Recall = %.2f%%" % (recall * 100))
