import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.dataloader import DataGenerator, detection_collate
from utils.utils import get_classes
from torch import nn
from tqdm import tqdm
import time
import datetime
import numpy as np

from .utils import get_lr


def fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch,
                  cuda, backbone, aa, gen_test):
    total_loss = 0
    total_accuracy = 0
    total_recall = 0
    val_loss = 0

    a = torch.Tensor(np.array(1e-7))
    total = len(gen_test)
    eval_pre = 0
    eval_recall = 0

    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            images, targets = batch
            with torch.no_grad():
                images = torch.from_numpy(images).type(torch.FloatTensor)
                targets = torch.from_numpy(targets).type(torch.FloatTensor)
                if cuda:
                    images = images.cuda()
                    targets = targets.cuda()

            optimizer.zero_grad()
            outputs = model_train(images)

            # outputs = outputs.sigmoid()
            # loss_value = nn.CrossEntropyLoss()(outputs, targets)

            loss_value = nn.BCELoss()(outputs.sigmoid(), targets)
            loss_value.backward()
            optimizer.step()

            total_loss += loss_value.item()

            with torch.no_grad():
                accuracy_th = 0.5
                pred_result = outputs.sigmoid() > accuracy_th
                pred_result = pred_result.float()
                pred_one_num = torch.sum(pred_result)

                target_one_num = torch.sum(targets)
                true_predict_num = torch.sum(pred_result * targets)
                # 模型预测的结果中有多少个是正确的
                precision = true_predict_num / (pred_one_num + a)
                # 模型预测正确的结果中，占所有真实标签的数量
                recall = true_predict_num / target_one_num

                total_accuracy += precision.item()
                total_recall += recall.item()

            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'accuracy': total_accuracy / (iteration + 1),
                                'recall': total_recall / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')

    # if epoch % 10 == 0:
    model_train.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, targets = batch
            with torch.no_grad():
                images = torch.from_numpy(images).type(torch.FloatTensor)
                targets = torch.from_numpy(targets).type(torch.FloatTensor)
                if cuda:
                    images = images.cuda()
                    targets = targets.cuda()

                optimizer.zero_grad()

                outputs = model_train(images)
                # outputs = outputs.sigmoid()
                # loss_value = nn.CrossEntropyLoss()(outputs, targets)
                loss_value = nn.BCELoss()(outputs.sigmoid(), targets)
                val_loss += loss_value.item()

            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    loss_history.append_loss(total_loss / epoch_step, val_loss / epoch_step_val)
    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
    torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f-val_loss%.3f-' %
               ((epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val) + backbone + '-' + aa + '.pth')

    print('start test!')
    start_time = time.time()

    model_train.eval()
    if epoch % 1 == 0:
        for iteration, batch in enumerate(gen_test):
            if iteration >= epoch_step:
                break
            images, targets = batch

            with torch.no_grad():
                images = torch.from_numpy(images).type(torch.FloatTensor)
                targets = torch.from_numpy(targets).type(torch.FloatTensor)
                if cuda:
                    images = images.cuda()
                    targets = targets.cuda()

            with torch.no_grad():
                outputs = model_train(images)

                accuracy_th = 0.5
                pred_result = outputs.sigmoid() > accuracy_th
                pred_result = pred_result.float()
                pred_one_num = torch.sum(pred_result)

                target_one_num = torch.sum(targets)
                true_predict_num = torch.sum(pred_result * targets)
                # 模型预测的结果中有多少个是正确的
                precision = true_predict_num / (pred_one_num + a)
                # 模型预测正确的结果中，占所有真实标签的数量
                recall = true_predict_num / target_one_num

                eval_pre += precision.item()
                eval_recall += recall.item()
        elapsed = round(time.time() - start_time)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("======> precision = %.2f%%\trecall = %.2f%%" % (eval_pre / total * 100, eval_recall / total * 100))

        with open("/data/CNXA_COCO/evaluate_result/{}{}.txt".format(backbone, aa), "a") as f:
            f.write("In epoch{}, evaluate {}{} with spend time:{} =====> precision:{}\n".format(epoch + 1, backbone, aa,
                                                                                                elapsed,
                                                                                                eval_pre / total))
            f.close()