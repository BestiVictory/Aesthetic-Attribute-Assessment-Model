# -*- coding: utf-8 -*-
#import dataset2 as mydataset
import pandas as pd
import mindspore as ms
from mindspore import nn
from mindspore import ops as P
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV
from mindspore import context , Model ,Tensor
from efficientnet import efficientnet_b0, efficientnet_b1
from mindspore.nn import Accuracy
from mindspore import dtype as mstype
context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
from PIL import Image
from efficientnet import GenEfficientNet,efficientnet_b0
import numpy as np
import pooling

# 定义神经网络
class ECA_Module(nn.Cell):
    def __init__(self, channel, k_size=3):
        super(ECA_Module, self).__init__()
        self.avg_pool = pooling.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2,pad_mode="pad" ,has_bias=False)
        self.sigmoid = nn.Sigmoid()
        self.unsqueeze= P.ExpandDims()
    def construct(self, x):
        # x: input features with shape [b, c, h, w]
        #b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(3).transpose(0,2,1)).transpose(0,2,1)
        y = self.unsqueeze(y,2)

        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x) 
class Net(nn.Cell):
    def __init__(self, attributes_num=10):
        super(Net, self).__init__()
        norm_layer = nn.BatchNorm2d
        in_channels = 1280#1792#2048
        inter_channels = in_channels // 4
        number =in_channels
        k_size = 3
        
        self.attributes_num = attributes_num
        #self.pretrained = efficientnet_b0(attributes_num)
        self.eca1 = ECA_Module(in_channels) #class
        self.eca2 = ECA_Module(in_channels) #score
        self.eca3 = ECA_Module(in_channels) #light
        self.eca4 = ECA_Module(in_channels) #color
        self.eca5 = ECA_Module(in_channels) #composition

        
        self.classification1 = nn.SequentialCell(
            nn.Dropout(0.5),
            nn.Dense(number, self.attributes_num),
            nn.BatchNorm1d(self.attributes_num)
        )
        self.regression_score1 = nn.SequentialCell(
            nn.Dropout(0.5),
            nn.Dense(number, self.attributes_num),
            nn.BatchNorm1d(self.attributes_num)
        )
        self.regression_score2 = nn.Dense(self.attributes_num * 4, 1)

        self.regression_light1 = nn.SequentialCell(
            nn.Dropout(0.5),
            nn.Dense(number, self.attributes_num),
            nn.BatchNorm1d(self.attributes_num)
        )
        self.regression_light2 = nn.Dense(self.attributes_num + 4, 1)
        
        self.regression_color1 = nn.SequentialCell(
            nn.Dropout(0.5),
            nn.Dense(number, self.attributes_num),
            nn.BatchNorm1d(self.attributes_num)
        )
        self.regression_color2 = nn.Dense(self.attributes_num + 7, 1)
        
        self.regression_composition1 = nn.SequentialCell(
            nn.Dropout(0.5),
            nn.Dense(number, self.attributes_num),
            nn.BatchNorm1d(self.attributes_num)
        )
        self.regression_composition2 = nn.Dense(self.attributes_num + 10, 1)
        
        
        self.cast = P.Cast()
        self.relu = P.ReLU()
        self.adaptive_avg_pool2d = pooling.AdaptiveAvgPool2d((1,1))
        self.flatten = P.Flatten()
        self.cat = P.Concat(axis=1)
        self.log_softmax = nn.LogSoftmax(1)
        


        
    def construct(self, x,light,color,composition):
        #x = self.pretrained(x,w,h)
        x1 = self.eca1(x)
        x2 = self.eca2(x)
        x3 = self.eca3(x)
        x4 = self.eca4(x)
        x5 = self.eca5(x)
        
        out1 = self.relu(x1)
        out1 = self.adaptive_avg_pool2d(out1)
        out1 = self.flatten(out1)
        
        out2 = self.relu(x2)
        out2 = self.adaptive_avg_pool2d(out2)
        out2 = self.flatten(out2)

        out_light = self.relu(x3)
        out_light = self.adaptive_avg_pool2d(out_light)
        out_light = self.flatten(out_light)

        out_color = self.relu(x4)
        out_color = self.adaptive_avg_pool2d(out_color)
        out_color = self.flatten(out_color)
        
        out_composition = self.relu(x5)
        out_composition = self.adaptive_avg_pool2d(out_composition)
        out_composition = self.flatten(out_composition)
        
        category = self.classification1(out1)
        out2 = self.regression_score1(out2)
        out_light = self.regression_light1(out_light)
        out_color = self.regression_color1(out_color)
        out_composition = self.regression_composition1(out_composition)
        
        light = self.cast(light,ms.float32)
        color = self.cast(color,ms.float32)
        composition = self.cast(composition,ms.float32)
        
        out_score = self.cat((out2, out_light, out_color, out_composition))
        out_light = self.cat((out_light, light))
        out_color = self.cat((out_color, color))
        out_composition = self.cat((out_composition, composition))
        
        out_score = self.regression_score2(out_score)
        out_light = self.regression_light2(out_light)
        out_color = self.regression_color2(out_color)
        out_composition = self.regression_composition2(out_composition)
        
        return self.log_softmax(category),self.log_softmax(out2), out_light, out_color, out_composition, out_score

if __name__ == "__main__":
    output = Tensor(np.random.randint(-255, 255, [1, 3, 224, 224]), ms.float32)
    print (output.shape)
    model = Model(Net(10))
    #model =  Model(hcr.Net(2))
    predict_map , score= model.predict(output,224,224)
    print(predict_map)
    print(score)
    #print(predict_map.shape)