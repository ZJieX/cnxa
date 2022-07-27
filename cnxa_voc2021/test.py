from PIL import Image
import shutil
import os
import glob

Image.MAX_IMAGE_PIXELS = 2300000000
# import torch
#
# torch.manual_seed(1)
#
# a = torch.randint(0, 2, [2, 1, 21])
# b = torch.randint(0, 2, [2, 1, 21])
#
# print("a:{}, b:{}".format(a, b))
#
# print((a == b).sum(1))



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


if __name__ =='__main__':
    path = 'D://PyCharm/xiaozhijie/CNXA/CNX_V3/dataset/VOCdevkit/VOC2021/JPEGImages'
    for fname in os.listdir(path):
        print(fname)
        path_ = os.path.join(path, fname)

        try:
            img = Image.open(path_)
            new = letterbox_image(img, (300, 300))
        except:
            print('corrupt img', path_)
            shutil.move(path_, 'D://PyCharm/xiaozhijie/CNXA/CNX_V3/dataset')