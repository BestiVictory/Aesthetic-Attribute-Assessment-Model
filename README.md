# Aesthetic-Attribute-Assessment-Model
![image](./Network structure.pdf)

[Aesthetic Attribute Assessment of Images Numerically on Mixed Multi-attribute Datasets](https://arxiv.org/abs/2207.01806v1)  
Xin Jin, Xinning Li, Hao Lou, Chenyu Fan, Qiang Deng, Chaoen Xiao, Shuai Cui, Amit Kumar Singh
ACM Transactions on Multimedia Computing Communications and Applications (TOMM)

In this paper, we propose an aesthetic attribute assessment of images numerically on mixed multi-attribute datasets.We construct an image attribute dataset and introduce external features of attribute based on the dataset.We use EfficentNet-B0 to build a main network for extracting features of inputting images; then, we build ten-classification sub-networks behind the main network.We build three attribute-regression subnetworks and total-score-regression sub-networks behind the main network. Every ECA channel attention is added in each specific branch network in the sub-network.We applied the idea of teacher-student network in the total-score-regression branch-network, and guided the regression of the total score through attribute regression.When the three attribute-regression sub-networks are trained, we connect features values of external attribute, stored in the database, and neurons for the following training.

## Prerequisites 
* numpy==1.21.6
* pandas==1.3.5
* Pillow==9.0.1
* mindspore-gpu==1.6.0
* python==3.6.0

## Getting Started
1. Clone this repo:  
```
   git clone https://github.com/BestiVictory/HistoryNet.git  
   cd SOURCE_HistoryNet
``` 
2. Create a Virtual Environment  
```
   conda create -n HistoryNet python=3.6 
   conda activate HistoryNet
```
3. Install all the dependencies  
```
   pip install -r requirements.txt
```
## Dataset
The dataset can be download [here](https://github.com/BestiVictory/AMD-A).
You can also choose your own dataset.

## Model
1. Download it from [BaiduCloud](https://pan.baidu.com/s/1KQnVA77EBF3huCwG4dVsHQ) (code: j0gi)  
2. Now the model should be placed in `MODEL` in the root directory.


## Colorize Images
```
cd SOURCE_HistoryNet
python colorization.py
```
## Train 
```
cd SOURCE_HistoryNet
python HistoryNet.py
```

## Citation
If you find our code/models useful, please consider citing our paper: 
```
@article{jin2022aesthetic,
  title={Aesthetic Attribute Assessment of Images Numerically on Mixed Multi-attribute Datasets},
  author={Xin Jin, Xinning Li, Hao Lou, Chenyu Fan, Qiang Deng, Chaoen Xiao, Shuai Cui, Amit Kumar Singh},
  journal={arXiv preprint arXiv:2207.01806},
  year={2022}
}
```
