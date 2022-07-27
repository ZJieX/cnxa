# pytorch实现现阶段主流分类网络


### 目录
1. [所需环境 Environment](#所需环境)
2. [文件下载 Download](#文件下载)
3. [训练步骤 How2train](#训练步骤)
4. [预测步骤 How2predict](#预测步骤)
5. [评估步骤 How2eval](#评估步骤)

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

[comment]: <> (模型文件&#40;loss与val_loss都取最小&#41;)


[comment]: <> (| name | resolution |top1 |top5 |#params | FLOPs | model |)

[comment]: <> (|:---:|:---:|:---:|:---:|:---:| :---:|:---:|)

[comment]: <> (| CNX | 224x224 | 89.46% | 97.14% | 87.163M| 15.359G| [model]&#40;https://pan.baidu.com/s/1rISHEiEDD82rcuH_OHd98Q?pwd=8pk3&#41;|)

[comment]: <> (| ViT | 224x224 | 89.33% | 98.00% | 85.723M | 16.856G | [model]&#40;https://pan.baidu.com/s/1YvUL3JueJD7eD3smCEVbCQ?pwd=qpy6&#41;|)

[comment]: <> (| CNXAECANET | 224x224 | 90.34% | 97.32%| 87.613M| 15.359G| [model]&#40;https://pan.baidu.com/s/1K5yD4YACB4r_EJ-_hsRElQ?pwd=mct3&#41;|)

[comment]: <> (| CNXASENET | 224x224 | 89.68% | 97.31% |87.745M | 15.359G|[model]&#40;https://pan.baidu.com/s/1XTsPJahl3juTzIK0gHvctg?pwd=4s2v&#41;|)

[comment]: <> (| CNXACA | 224x224 | 89.89% | 97.29%|87.745M | 15.360G|[model]&#40;https://pan.baidu.com/s/12v6FY2rZT6gD99CvWwfrFg?pwd=hi7e&#41;|)

[comment]: <> (| CNXASA | 224x224 | 89.86% | 97.19%|87.613M| 15.359G|[model]&#40;https://pan.baidu.com/s/1RFgCeTEzfhJkmbD_-3O-iw?pwd=cqhx&#41;|)

[comment]: <> (| CNXACASA | 224x224 | 90.24% | 97.64%|87.745M| 15.360G|[model]&#40;https://pan.baidu.com/s/1SSL9kFyF-Vmqidbf8tlzvA?pwd=cxqp&#41;|)

[comment]: <> (| CNXASACA | 224x224 | 90.24% | 97.64%|87.745M| 15.360G|model|)

[comment]: <> (| CNXABK | 224x224 | 88.96% | 97.49%|246.867M| 16.097G|[model]&#40;https://pan.baidu.com/s/1QkubPXYbczo4gnoa-yWsqw?pwd=90b9&#41;|)

[comment]: <> (模型文件&#40;只取val_loss都取最小&#41;)

| name | resolution |top1 |top5 |#params | FLOPs | model |
|:---:|:---:|:---:|:---:|:---:| :---:|:---:|
| CNX | 224x224 | 97.43% | 99.77% | 87.163M| 15.359G|[model](https://pan.baidu.com/s/1M3G3pn1_NB5VXUWFKFayyA?pwd=tqsv)|
| ViT | 224x224 | 89.94% | 98.33%| 85.723M | 16.856G |[model](https://pan.baidu.com/s/1n0d4ZfIutIxmwtTC2mCnIg?pwd=vlpa)|
| CNXAECANET | 224x224 | 97.68% | 99.78% | 87.613M| 15.359G|[model](https://pan.baidu.com/s/1ri0RYfotyXXzVaJ2Dsxavg?pwd=i17o)|
| CNXASENET | 224x224 | 97.50% | 99.79% | 87.745M | 15.359G|[model](https://pan.baidu.com/s/11MPIS1lbTQzjGgHw0SZfcg?pwd=sjq8)|
| CNXACA | 224x224 | 97.69% |99.69% |87.745M | 15.360G|[model](https://pan.baidu.com/s/1F4pztz7J8qddG3D8KiIWaQ?pwd=q2vm)|
| CNXASA | 224x224 | 97.46% |99.71% |87.613M| 15.359G|[model](https://pan.baidu.com/s/1eTv1HToevL8gRFk2lruegg?pwd=xn7w)|
| CNXACASA | 224x224 | 97.51% | 99.70%|87.745M| 15.360G|[model](https://pan.baidu.com/s/1xAf3MQTngkdJM4GX9_Sswg?pwd=4xd8)|
| CNXASACA | 224x224 | 97.48% | 99.60%|87.745M| 15.360G|[model](https://pan.baidu.com/s/1sJNObBShOHE1P-ndEQO79w?pwd=t2u8)|
| CNXABK | 224x224 | 97.84% | 99.72%|227.207M| 26.975G|[model](https://pan.baidu.com/s/18-ItUitnUWSfmU_QgZrUow?pwd=st3v)|
| CNXABK_DW | 224x224 | 97.73% | 99.78%|98.239M| 16.543G|[model](https://pan.baidu.com/s/1B8i4Cdn623vl33BM43RM6w?pwd=tk2s)|

数据集ImageNet100下载请[点击](https://pan.baidu.com/s/1KbRhZrV2xjGWMUjw0yTXIQ?pwd=pcai 
) 提取码：pcai

### 训练步骤
1. 准备数据集(ImageNet100)，格式如下：
```
|-datasets
    |-xxx
        |-classes_one
            |-123.jpg
            |-234.jpg
        |-classes_two
            |-345.jpg
            |-456.jpg
        |-...
```
修改 create_txt.py中的root=" "；

运行 python create_txt.py生成cls_test.txt用于评估，cls_train.txt用于训练。

2. 修改train.py中的classes_path、backbone、aa、model_path以及一些超参数。
运行 python train.py 进行训练。

### 预测步骤
1. 在classification.py文件里面，在如下部分修改model_path、classes_path、backbone和alpha使其对应训练好的文件；**model_path对应logs文件夹下面的训练好的模型，classes_path是model_path对应分的类，backbone对应使用的主干特征提取网络。
一般挑val_loss最小的进行预测，或者挑选loss和val_loss都是最小的进行预测。

2. 运行 python predict.py,在终端输入图片所在位置即可进行预测。

### 评估步骤
1. 由于已经生成cls_test.txt评估文件；
2. 之后在classification.py文件里面修改如下部分model_path、classes_path、backbone和alpha使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类，backbone对应使用的主干特征提取网络；
3. 运行python eval_top1.py 和 python eval_top5.py；
4. 精度结果生成在evaluate_result里面可自行查看。


