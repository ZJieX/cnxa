import numpy as np
import torch
import time
import datetime
from PIL import Image

from classification import (Classification, cvtColor,
                            letterbox_image, preprocess_input)
from utils.utils import letterbox_image
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


def evaluate_precision(classfication, lines):
    print("=======> Start calculating acc")
    total = len(lines)
    acc = list()
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
        
        y_true = labels.cpu().numpy()
        y_pred = pred.cpu().numpy()
        
        accuracy_th = 0.5
        pred_result = (1 / (1 + np.exp(-y_pred))) > accuracy_th
        
        pred_result = pred_result.astype(float)
        pred_one_num = np.sum(pred_result)

        # target_one_num = torch.sum(labels)
        true_predict_num = np.sum(pred_result * y_true)
        # 模型预测的结果中有多少个是正确的
        Acc = true_predict_num / (pred_one_num + a)

        acc.append(Acc)
        if index % 100 == 0:
            print("[%d/%d]" % (index, total))
    return np.mean(acc)


# def evaluate_f1(classfication, lines):
#     print("=======> Start calculating f1")
#     f1 = list()
#     precision = list()
#     recall = list()
#     total = len(lines)
#     a = 1e-9
#
#     for index, line in enumerate(lines):
#         annotation_path = line.split(';')[1].split()[0]
#         x = Image.open(annotation_path)
#         y = list(line.split(';')[0])[1: len(line.split(';')[0]) - 1]
#         y = y[0: len(y): 3]
#         labels = list(map(int, y))
#         labels = np.array(labels)
#         labels = torch.from_numpy(labels)
#
#         pred = classfication.detect_image(x)
#         pred = (pred > 0)
#
#         TP = np.sum((pred[0].cpu().numpy() == True) & (labels.cpu().numpy() == 1))
#         # print(TP)
#         FN = np.sum((pred[0].cpu().numpy() == False) & (labels.cpu().numpy() == 1))
#         # print(FN)
#         FP = np.sum((pred[0].cpu().numpy() == True) & (labels.cpu().numpy() == 0))
#         # print(FP)
#
#         precision.append(TP / (TP + FP + a))
#         recall.append(TP / (TP + FN + a))
#         f1.append((2 * precision[-1] * recall[-1]) / (precision[-1] + recall[-1] + a))
#         if index % 100 == 0:
#             print("[%d/%d]" % (index, total))
#
#     mf1 = sum(f1) / len(f1)
#     mprecision = sum(precision) / len(precision)
#     mrecall = sum(recall) / len(recall)
#
#     return mf1, mprecision, mrecall


classfication = classification()
with open("./cls_test_voc2021.txt", "r") as f:
    lines = f.readlines()

start_time = time.time()
backbone = classification.get_defaults("backbone")
aa = classification.get_defaults("aa")
model = classification.get_defaults("model_path")

acc = evaluate_precision(classfication, lines)

elapsed = round(time.time() - start_time)
elapsed = str(datetime.timedelta(seconds=elapsed))

with open("evaluate_result/{}{}.txt".format(backbone, aa), "a") as f:
    f.write("\n{} is being used for evaluation"
            "evaluate {}{} with spend time:{} =====> Acc:{}".format(model, backbone, aa, elapsed, acc * 100))
    f.close()

print("Acc = %.2f%%" % (acc * 100))
