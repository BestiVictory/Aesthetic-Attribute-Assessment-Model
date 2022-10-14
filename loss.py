# -*- coding: utf-8 -*-
#import dataset_c_r as mydataset
import pandas as pd
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import ops as P
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV
from mindspore import context , Model , Tensor
from mindspore.nn import Accuracy
from mindspore import dtype as mstype
context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

data_path = "./images/"  
lable_path = "./trainlist_c_test.csv"

class My_loss(nn.Cell):
    def __init__(self):
        super(My_loss, self).__init__()
        self.critirion_class = P.NLLLoss(reduction='none')
        self.critirion_second_class = P.NLLLoss(reduction='none')
        self.mul = P.Mul()
        self.mean = P.ReduceMean()
        #self.label = label
        #self.second_label = second_label
        #self.legth = legth
        #self.output_class = output_class
        self.cast = P.Cast()
        self.weight = Tensor(np.ones(10), ms.float32)
        
    def construct(self,label,second_label,length,output_class):
        #loss_class = self.critirion_class(self.cast(output_class,ms.float32), self.cast(label,ms.int32),self.weight)
        #loss_second_class = self.critirion_second_class(self.cast(output_class,ms.float32), self.cast(second_label,ms.int32),self.weight)

        loss_class ,weight= self.critirion_class(output_class, self.cast(label,ms.int32), self.weight)
        loss_second_class ,weight= self.critirion_second_class(output_class,self.cast(second_label,ms.int32),self.weight)
        loss_second_class = self.mul(loss_second_class,self.cast(length,ms.float32))  #loss_class
        loss = loss_class + loss_second_class
        
        return self.mean(loss)
        
class rank_loss(nn.Cell):
    def __init__(self):
        super(rank_loss, self).__init__()
        self.gt = ops.Greater()
        self.zeroslike = ops.ZerosLike()
        self.cast = P.Cast()
        self.sum = P.ReduceSum()
        
    def construct(self, regre, target):
        loss = 0
        for i in range(len(target)):
            reg1 = regre[:i]
            reg2 = regre[len(target)-i:]
            tar1 = target[:i]
            tar2 = target[len(target)-i:]

            tar = self.gt(tar1,tar2)
            tar = (tar*2)-1
            reg = reg1-reg2
            alpha = 0
            zero = self.zeroslike(tar)
            zero = self.cast(zero,ms.float32)
            
            _max = P.ArgMaxWithValue(alpha-(tar*reg))
            a , b = _max(zero)
            loss += self.sum(b,a)        

        return 2*loss/(len(target)*(len(target)-1))
        
class Soft_loss(nn.Cell):
    def __init__(self):
        super(Soft_loss, self).__init__()
        self.critirion_class = P.KLDivLoss(reduction='none')
        self.mean = P.ReduceMean()

        
    def construct(self,output_class, output_class2):
        loss_class = self.critirion_class(output_class2, output_class)
        loss = loss_class 
        
        return self.mean(loss)
        
if __name__ == "__main__":
    m = nn.LogSoftmax(axis=1)
    loss = My_loss()
    input = Tensor(np.random.randn(3, 2), ms.float32)
    labels = Tensor([1, 0, 4], ms.int32)
    length = Tensor([1.0, 0, 4.0], ms.float32)


    train_transforms = ds.transforms.c_transforms.Compose([
       
        ds.vision.c_transforms.RandomHorizontalFlip(),
        #ds.vision.py_transforms.ToTensor(),
        ds.vision.c_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ds.vision.c_transforms.HWC2CHW()
    ])

    ava_train = mydataset.GetDatasetGenerator(data_path, lable_path, divide=0.5, transforms=train_transforms,ranges=0.05)
    train_dataset = ds.GeneratorDataset(ava_train, ["data", "w", "h", "label" , "second_label","length", "target"], shuffle=False)
    train_dataset = train_dataset.batch(3)
    for data in train_dataset.create_dict_iterator():
        loss = loss(data["label"],data["second_label"],data["legth"],m(input))
        print(loss)
