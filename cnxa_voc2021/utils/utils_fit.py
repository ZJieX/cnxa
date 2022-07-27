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

from .utils import get_lr, compute_mAP, eval_map


def fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch,
                  cuda, backbone, aa, gen_test):
    total_loss = 0
    val_loss = 0
    # scaler = torch.cuda.amp.GradScaler()
    # autocast = torch.cuda.amp.autocast
    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        mAP = []
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
            # with autocast():
            outputs = model_train(images)

            mAP.append(compute_mAP(targets, outputs, cuda))
            loss_value = nn.MultiLabelSoftMarginLoss()(outputs, targets)

            # scaler.scale(loss_value).backward()

            # scaler.step(optimizer)

            # scaler.update()
            loss_value.backward()
            optimizer.step()

            total_loss += loss_value.item()

            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'lr': get_lr(optimizer),
                                'mAP': 100 * np.mean(mAP[-21:])})
            pbar.update(1)

    print('Finish Train')

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

                loss_value = nn.MultiLabelSoftMarginLoss()(outputs, targets)
                val_loss += loss_value.item()

            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    loss_history.append_loss(total_loss / epoch_step, val_loss / epoch_step_val)
    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
    torch.save(model.state_dict(), 'logs/CNXABK_DW/avg/ep%03d-loss%.3f-val_loss%.3f-' %
               ((epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val) + backbone + '-' + aa + '.pth')

    print('start test!')
    start_time = time.time()

    if epoch % 1 == 0:
        mAP = []
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

                mAP.append(compute_mAP(targets, outputs))

        elapsed = round(time.time() - start_time)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('TESTING: %d), mAP %.2f%%' % (epoch + 1, 100 * np.mean(mAP)))

        with open("/data/CNXA_VOC2021/evaluate_result/{}{}_mAP.txt".format(backbone, aa), "a") as f:
            f.write("In epoch{}, evaluate {}{} with spend time:{} =====> mAP:{}\n".format(epoch + 1, backbone, aa,
                                                                                                elapsed,
                                                                                                100 * np.mean(mAP)))
            f.close()
