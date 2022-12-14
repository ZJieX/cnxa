# CNXA
Now there are only V1 and V2 versions. The final version will be completed soon and will be uploaded soon. Adding various attentional mechanism networks to the work of the 2022 ConvNeXt-base network resulted in a new network backbone with 0.5 ± 0.3 accuracy over the ImageNet100 dataset compared to the original network.
# Pytorch implements the current mainstream classification network + attention mechanism


### Directory
1. [Environment](#The required environment)
2. [Download](#File download)
3. [How2train](#The training steps)
4. [How2predict](#Prediction steps)
5. [How2eval](#Assessment steps)

### The required environment
pytorch >= 1.12.0 + CUDA >= 11.0

Docker environment is recommended to run on Linux, docker image based on this project please [click](https://pan.baidu.com/s/1963dCct6ZERe2PB1vVDohQ?pwd=lgbd 
), Extract the code: lgbd. download, this step must understand some Linux command has been based on docker.
######Using the Docker steps (docker is already installed on Linux by default) :
    1. Download the docker image (CNXA_V3.tar) provided by XFPT or winSCP or other software or technology into Linux;
    2. Run the docker load -i CNXA_V3.tar command to import the image.
    3. After the import is complete, enter Docker images to view;
    4. Pass this project's Code folder to Linux as well, such as /home/code/;
    5. On Linux, go to the /home/code directory;
    6. Command line input:docker run -it --rm --gpus all --shm-size 6g -v $PWD:/data/ cnx_v3:latest /bin/bash; 
    7. cd /data/ to enter the project directory, please refer to the following operation steps;
    8. If you are a Docker master, please set your own parameters and map to docker inside.

### File download

Model file (val_loss)

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

Dataset ImageNet100 download please [click](https://pan.baidu.com/s/1F0IsfMicGg3h3Prrz6sDXg?pwd=t6uf 
), Extract the code: t6uf.

### The training steps
1. Prepare the data set (ImageNet100) in the following format:
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
Change root=" " in create_txt.py;

Run python create_txt.py to generate cls_test.txt for evaluation and cls_train.txt for training.

2. Modify classes_path, backbone, aa, model_path and some super parameters in train.py.
Run Python train.py to train.

### Prediction steps
1. In the classification.py file, modify model_path, classes_path, backbone and aa in the following sections to make them correspond to trained files; **model_path corresponds to the trained model under the logs folder. Classes_path is the class that model_path should be divided into. Backbone corresponds to the feature extraction network;
Generally, the least val_loss is selected for prediction, or both loss and val_loss are selected for prediction.

2. Run python predict.py and enter the image location on the terminal to predict the image.

### Assessment steps
1. The cls_test.txt evaluation file has been generated;
2. Then modify the following parts of model_path, classes_path, backbone and aa in the classification.py file to make them correspond to the trained file; **model_path corresponds to the weight file under logs folder. Classes_path is the class that model_path should be divided into. Backbone corresponds to the feature extraction network;
Generally, the least val_loss is selected for prediction, or both loss and val_loss are selected for prediction.

3. Run python eval_top1.py and python eval_top5.py;
4. Accurate results are generated in evaluate_result for self-review.