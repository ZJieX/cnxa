# ReID, 代码参照浙江大学罗浩博士[行人重识别项目](https://github.com/michuanhaohao/)https://github.com/michuanhaohao/)


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
#####Market1501数据集评估结果:
| model |  mAP | Rank@1 | Rank@5 |  model |
|:---:|:---:|:---:|:---:|:---:|
| CNX | 79.5% | 92.1% | 97.6% | [model](https://pan.baidu.com/s/1eiIw7vzk8SlT3uFbA9ANzQ?pwd=1vw1) |
| CBN | 77.3% | 91.3% | 97.1% | model |
| CNXACA | 79.4% | 92.0% | 97.5% | [model](https://pan.baidu.com/s/1W-9rp_EVhvN7wsA7hB47Ng?pwd=weqb) |
| CNXASA | 79.4% | 91.9% | 97.6% | [model](https://pan.baidu.com/s/1ijhj7C85O7rRVviyfAF5_w?pwd=v0tk) |
| CNXABK_DW | 79.5% | 92.4% | 97.4% | [model](https://pan.baidu.com/s/1CChlgdPON18eiX9jVjoTeA?pwd=5e1o) |

#####DukeMTMC-reID数据集评估结果:
| model |  mAP | Rank@1 | Rank@5 |  model |
|:---:|:---:|:---:|:---:|:---:|
| CNX | 69.3% | 83.8% | 92.8% | [model](https://pan.baidu.com/s/1FS3jyJHeTSzIUWS-mq-FpA?pwd=7k94) |
| CBN | 67.3% | 82.5% | 91.7% | model |
| CNXACA | 70.4% | 84.0% | 93.0% | [model](https://pan.baidu.com/s/1-0t6IhfWOZec2qWQPz5KIQ?pwd=igmo) |
| CNXASA | 70.1% | 84.0% | 93.2% | [model](https://pan.baidu.com/s/11mZKHqevofvFpvBRECHqSg?pwd=ndrj) |
| CNXABK_DW | 70.3% | 84.4% | 93.1% | [model](https://pan.baidu.com/s/1OT18VRPLEBf1xr9lgZ9u5A?pwd=0rpz) |


### 训练评估步骤
1. 准备数据集(Market1501)，格式如下：
```
|-dataset
    |-maeket1501
        |-bounding_box_test
        |-bounding_box_train
        |-query
        |- ...
        
```

2. 准备数据集(DukeMTMC-reID)，格式如下：
```
|-dataset
    |-dukemtmc-reid
        |-DukeMTMC-reID
            |-bounding_box_test
            |-bounding_box_train
            |-query
            |- ...
        
```
3. 训练步骤：
```
    ①. 修改Train_duke.sh或者Train_market.sh中的 MODEL.NAME "('convnext')";
        convnext代表CNX, convnextaa代表三种两种注意力机制，convbk代表我们的大核方法;
    ②. 在使用convnextaa时，需要进入modeling/baseline.py， 修改aa='ca'或者aa='sa';
    ③. 下载好ConvNext-Small网络的预训练模型放置weights文件夹中;
    ④. 在终端输入 sh Train_market.sh 或者 sh Train_duke.sh 训练两个不同的数据集。
```

4. 评估步骤：
```
    ①. 修改Test.sh或者Train_market.md中的 MODEL.NAME "('convnext')";
        convnext代表CNX, convnextaa代表三种两种注意力机制，convbk代表我们的大核方法;
    ②. 在使用convnextaa时，需要进入modeling/baseline.py， 修改aa='ca'或者aa='sa';
    ③. 修改Test.sh中的 DATASETS.NAMES "('market1501')" 选择评估哪个数据集，为market1501，dukemtmc
    ④. 修改 TEST.WEIGHT 定位到训练好的模型权重路径
    ⑤. 在终端输入 sh Test.sh 进行评估
```



