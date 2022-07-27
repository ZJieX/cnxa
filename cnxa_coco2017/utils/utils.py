import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score
import os


# models = {
#     'resnet18': resnet18,
#     'resnet34': resnet34,
#     'resnet50': resnet50,
#     'resnet101': resnet101,
#     'resnet152': resnet152
# }

# ---------------------------------------------------#
#   对输入图像进行resize
# ---------------------------------------------------#
def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    h, w = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


# ---------------------------------------------------#
#   获得类
# ---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


# ----------------------------------------#
#   预处理训练图片
# ----------------------------------------#
def preprocess_input(x):
    x /= 127.5
    x -= 1.
    return x


# ---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
# ---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[-2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

    # ---------------------------------------------------#


#   获得学习率
# ---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)


def compute_mAP(labels, outputs):
    y_true = labels.data.cpu().numpy()
    y_pred = outputs.data.cpu().numpy()
    AP = []
    for i in range(y_true.shape[0]):
        AP.append(average_precision_score(y_true[i], y_pred[i]))
    return np.mean(AP)


def eval_map(net, logger, val_loader, steps, gpu, crops):
    if gpu is not None:
        net.cuda()
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    mAP = []
    net.eval()
    for i, (images, labels) in enumerate(val_loader):
        images = images.view((-1, 3, 224, 224))
        if gpu is not None:
            images = images.cuda()

        # Forward + Backward + Optimize
        outputs = net(images)
        outputs = outputs.cpu().data
        if crops != 0:
            outputs = outputs.view((-1, crops, 20))
            outputs = outputs.mean(dim=1).view((-1, 20))
        else:
            outputs = outputs.view((-1, 20))

        # score = tnt.meter.mAPMeter(outputs, labels)
        mAP.append(compute_mAP(labels, outputs))

    if logger is not None:
        logger.scalar_summary('mAP', np.mean(mAP), steps)
    print('TESTING: %d), mAP %.2f%%' % (steps, 100 * np.mean(mAP)))
    net.train()


# def eval_macc(val_loader, model_path="../checkpoints/resnet18_190515_2049_001.pth",
#               model="resnet18", gpu=None, crops=0):
#     """
#     Evaluate a model on a dataset, using mAcc as index
#     :param val_loader: the dataloader(torch.utils.dataloader) object
#     :param model_path: the path to the model
#     :param model: which kind is the model
#     :param gpu: which gpu to use
#     :param crops: how many random crops
#     :return: mAcc on the dataset
#     """
#     net = load_model_from_file(model_path, model=model, load_fc=True)
#     if gpu is not None:
#         net.cuda()
#         os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#         os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
#
#     acc = []
#     net.eval()
#     for idx, (images, labels) in enumerate(val_loader):
#         images = images.view((-1, 3, 224, 224))
#         if gpu is not None:
#             images = images.cuda()
#
#         outputs = net(images)
#         outputs = outputs.cpu().data
#         if crops != 0:
#             outputs = outputs.view((-1, crops, 20))
#             outputs = outputs.max(dim=1)[0].view((-1, 20))
#         else:
#             outputs = outputs.view((-1, 20))
#
#         # outputs: shape [batchsize * num_classes]
#         outputs = (outputs > 0)
#         acc.append(np.sum((outputs.numpy() == labels.numpy()).astype(float)) / (val_loader.batch_size * 20))
#
#         print("Evaluating mAcc, Batch_size: %d" % idx, end='\r')
#
#     macc = sum(acc) / len(acc)
#     print("\nFinal mAcc: %f" % macc)
#     return macc
#
#
# def eval_wacc(val_loader, model_path="../checkpoints/resnet18_190515_2049_001.pth",
#               model="resnet18", gpu=None, crops=0):
#     """
#     Evaluate a model on a dataset, using wAcc as index
#     :param val_loader: the dataloader(torch.utils.dataloader) object
#     :param model_path: the path to the model
#     :param model: which kind is the model
#     :param gpu: which gpu to use
#     :param crops: how many random crops
#     :return: mAcc on the dataset
#     """
#     net = load_model_from_file(model_path, model=model, load_fc=True)
#     if gpu is not None:
#         net.cuda()
#         os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#         os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
#
#     acc = np.zeros(20)
#     net.eval()
#     freq = np.zeros(20)
#
#     for idx, (images, labels) in enumerate(val_loader):
#         # Frequency of the labels
#         freq += np.sum(labels.numpy(), axis=0)
#         images = images.view((-1, 3, 224, 224))
#
#         if gpu is not None:
#             images = images.cuda()
#         outputs = net(images)
#         outputs = outputs.cpu().data
#         if crops != 0:
#             outputs = outputs.view((-1, crops, 20))
#             outputs = outputs.max(dim=1)[0].view((-1, 20))
#         else:
#             outputs = outputs.view((-1, 20))
#         outputs = (outputs > 0)
#         acc += np.sum((outputs.numpy() == labels.numpy()), axis=0).astype(float)
#
#         print("Evaluating wAcc, Batch_size: %d" % idx, end="\r")
#
#     freq = freq / np.sum(freq)
#     acc = acc / len(val_loader.dataset)
#
#     wacc = np.dot(freq, acc)
#     print("\nFinal wAcc: %f" % wacc)
#     return wacc
#
#
# def eval_f1(val_loader, model_path="../checkpoints/resnet18_190515_2049_001.pth",
#             model="resnet18", gpu=None, crops=0):
#     """
#     Evaluate a model on a dataset, using f1 as index
#     :param val_loader: the dataloader(torch.utils.dataloader) object
#     :param model_path: the path to the model
#     :param model: which kind is the model
#     :param gpu: which gpu to use
#     :param crops: how many random crops
#     :return: f1 score on the dataset
#     """
#     net = load_model_from_file(model_path, model=model, load_fc=True)
#     if gpu is not None:
#         net.cuda()
#         os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#         os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
#
#     f1 = []
#     precision = []
#     recall = []
#     net.eval()
#     for idx, (images, labels) in enumerate(val_loader):
#         images = images.view((-1, 3, 224, 224))
#         if gpu is not None:
#             images = images.cuda()
#
#         outputs = net(images)
#         outputs = outputs.cpu().data
#         if crops != 0:
#             outputs = outputs.view((-1, crops, 20))
#             outputs = outputs.max(dim=1)[0].view((-1, 20))
#         else:
#             outputs = outputs.view((-1, 20))
#
#         # outputs: shape [batchsize * num_classes]
#         outputs = (outputs > 0)
#         TP = np.sum((outputs.numpy() == 1) & (labels.numpy() == 1))
#         # TN = np.sum((outputs.numpy() == 0) & (labels.numpy() == 0))
#         FN = np.sum((outputs.numpy() == 0) & (labels.numpy() == 1))
#         FP = np.sum((outputs.numpy() == 1) & (labels.numpy() == 0))
#         precision.append(TP / (TP + FP))
#         recall.append(TP / (TP + FN))
#         f1.append((2 * precision[-1] * recall[-1]) / (precision[-1] + recall[-1]))
#
#         print("Evaluating f1, Batch_size: %d" % idx, end='\r')
#
#     mf1 = sum(f1) / len(f1)
#     mprecision = sum(precision) / len(precision)
#     mrecall = sum(recall) / len(recall)
#     print("\nFinal f1-score: %f" % mf1)
#     print("precision: %f" % mprecision)
#     print("recall: %f" % mrecall)
#     return mf1
#
#
# def load_model_from_file(filepath, model="resnet18", load_fc=None):
#     """
#     Load the trained model from .pth file. Only for the same model trained before
#     :param filepath: the path to .pth file
#     :param model: the backbone network
#     :param load_fc: whether to load fc layer
#     :return: loaded model
#     """
#     # Get the initial network
#     dict_init = torch.load(filepath)
#     keys = [k for k, v in dict_init.items()]
#     keys.sort()
#     # Generate a new network
#     net = models[model](pretrained=False, num_classes=20)
#     model_dict = net.state_dict()
#     # load the layers
#     to_load = []
#     for k in keys:
#         if k not in model_dict:
#             continue
#         if load_fc is not None or 'fc' not in k:
#             to_load.append(k)
#     # load the dict
#     dict_init = {k: v for k, v in dict_init.items() if k in to_load and k in model_dict}
#     model_dict.update(dict_init)
#     net.load_state_dict(model_dict)
#
#     return net
