# pytorch实现现阶段主流分类网络


### 目录
1. [所需环境 Environment](#所需环境)
2. [文件下载 Download](#文件下载)
3. [训练步骤 How2train](#训练评估步骤)

### 所需环境
pytorch >= 1.12.0 + CUDA >= 11.0

推荐使用docker环境在linux上运行，基于此项目的docker镜像请[点击](https://pan.baidu.com/s/1963dCct6ZERe2PB1vVDohQ?pwd=lgbd
)下载. 提取码：lgbd。此步骤必须得了解些许linux命令已经docker基础。
######使用docker步骤(默认linux上已经安装docker)：
    1. 下载好提供的docker镜像(cnx_v3.tar)，通过xfpt或者winSCP等软件或技术传入到linux中；
    2. 命令行输入 docker load -i cnx_v3.tar，等待镜像导入完成；
    3. 导入完成后，输入docker images查看；
    4. 将此项目的代码文件夹也传入到linux中，比如传入到 /home/Code/中；
    5. 在linux中进入到 /home/Code目录；
    6. 命令行输入：docker run -it --rm --gpus all --shm-size 6g -v $PWD:/data/ cnx_v3:latest /bin/bash； 
    7. 进入后cd /data/进入到项目目录，至此运行步骤请参考以下内容；
    8. 如果有docker高手，请自行设置参数以及映射到docker里面。

### 文件下载
| model | resolution | mAP | mAcc | Precision | Recall | F1 | Para | FLOPs | model |
|:---:|:---:|:---:|:---:|:---:| :---:|:---:|:---:|:---:|:---:|
| CNX | 300x300 | 94.58% | 97.95% | 89.68% | 81.84% | 83.50% | 87.532M | 25.701G | [model](https://pan.baidu.com/s/1bcsu-43abh8Vd4hNO3_ZXg?pwd=kx51) |
| ViT | 300x300 | 78.74% | 95.78% | 66.46% | 60.35% | 61.09% | 85.663M | 27.808G | [model](https://pan.baidu.com/s/18-s7g6RJj5Cxx5cdWDcciQ?pwd=295) |
| CNXACA | 300x300 | 95.50% | 97.98% | 90.74% | 82.25% | 84.21% | 87.664M | 25.702G | [model](https://pan.baidu.com/s/16K5aHaDdVGeTA6aLWl7szQ?pwd=vh6z) |
| CNXASA | 300x300 | 94.85% | 97.92% | 89.91% | 80.39% | 82.80% | 87.532M | 25.701G | [model](https://pan.baidu.com/s/16YK3ARMJr5SJg04zPVkfqQ?pwd=m2x2) |
| CNXACASA | 300x300 | 94.60% | 97.90% | 89.83% | 79.92% | 82.58% | 87.532M | 25.702G | [model](https://pan.baidu.com/s/1a37lftxrnAsfAJGoIj5uJQ?pwd=0jyb) |
| CNXABK | 300x300 | 95.55% | 98.06% | 91.81% | 83.02% | 85.08% | 231.846M | 38.262G | [model](https://pan.baidu.com/s/1D547scFL96q72X_7YJOtMA?pwd=xo7b) |
| CNXABK_DW | 300x300 | 95.57% | 98.07% | 90.95% | 83.06% | 84.77% | 98.158M | 27.551G | [model](https://pan.baidu.com/s/1C0YBoSq4jzEft5saONmYrg?pwd=6in8) |


数据集VOC2021下载请[点击](https://pan.baidu.com/s/1d85TPuhMutvsHYyKeouoeQ?pwd=px81) 

### 训练评估步骤
1. 准备数据集(VOC2021)，格式如下：
```
|-datasets
    |-VOCdevkit
        |-VOC2021
            |-Annotations
                |-xxxx.xml
                    ...
            |-JPEGImages
                |-xxxx.jpg
                    ...

```
修改 create_txt.py中的anno='', img_path=''以及classes=''，修成成自己数据集的路径；
            
    ①. anno 代表标注文件的路径；
    ②. img_path 代表图片的路径；
    ③. classes 代表包含所有类别的.txt文件所在的路径。

运行 python create_txt.py成cls_test_voc2021.txt用于评估，cls_train_voc2021.txt用于训练。

2. 输入python train.py
    ```
    --backbone 可以有'cnxa'、'cnx'和'vit'
    --aa 当backbone为cnxa时，可以进行设置，有'ca'、'sa'、'casa'、'bk'以及'bk_dw'
    --pad 当aa选用bk或者bk_dw时，可以进行设置，有'zero'、'max'和'avg'
    ```
3. 修改classification.py文件中的 "backbone"、"aa"、"pad"以及"model_path"，跟第2步设置一样，这样才能验证具体的某一个模型。
4. 修改evaluate.pyw文件127行中的测试数据集的.txt文件，之后运行 python evaluate.py进行评估
