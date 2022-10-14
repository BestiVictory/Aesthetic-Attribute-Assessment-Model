# -*- coding: utf-8 -*-
import new_dataset_fusion as mydataset
import pandas as pd
import mindspore as ms
from mindspore import nn ,save_checkpoint,export, load_checkpoint, load_param_into_net,Parameter,Tensor
from mindspore import ops as P
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV
from mindspore import context , Model ,Tensor
from efficientnet import efficientnet_b0, efficientnet_b1
from mindspore.nn import Accuracy
from mindspore import dtype as mstype
context.set_context(mode=context.GRAPH_MODE, device_target='GPU', device_id=2)
from PIL import Image
from efficientnet import GenEfficientNet,efficientnet_b0
import numpy as np
from new_hcr_models_fusion import Net
from loss import My_loss
from mindspore.train.callback import LossMonitor , ModelCheckpoint, CheckpointConfig

import mindspore.nn as nn
from mindspore.train.callback import Callback

class EvalCallBack(Callback):
    def __init__(self, model, eval_dataset, eval_per_epoch, epoch_per_eval):
        self.model = model
        self.eval_dataset = eval_dataset
        self.eval_per_epoch = eval_per_epoch
        self.epoch_per_eval = epoch_per_eval

    def epoch_end(self, run_context):
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        if cur_epoch % self.eval_per_epoch == 0:
            acc = self.model.eval(self.eval_dataset, dataset_sink_mode=False)
            self.epoch_per_eval["epoch"].append(cur_epoch)
            self.epoch_per_eval["mse"].append(acc["MSE"])
            print(acc)


class CustomWithLossCell(nn.Cell):
    def __init__(self, backbone,net, loss_fn):
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self._net = net
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, data, label, second_label ,length, target, light, color, composition):
        output = self._backbone(data)
        output_class, output_class2, output_light, output_color, output_composition, output_score = self._net(output,light, color, composition)
        return self._loss_fn( output_composition, target)
    
class CustomWithEvalCell(nn.Cell):
    def __init__(self, backbone,net):
        super(CustomWithEvalCell, self).__init__(auto_prefix=False)
        self._net = net
        self._backbone = backbone
        
    def construct(self, data, label, second_label ,length, target, light, color, composition):
        output = self._backbone(data)
        output_class, output_class2, output_light, output_color, output_composition, output_score = self._net(output,light, color, composition)
        return output_composition, target

if __name__ == "__main__":
    
    data_path = "./images/"
    lable_path = "./datalist/train_attributes.csv"
    val_path = "./datalist/valid_attributes.csv"
    ckpt_save_dir = "./regre3"
    batch_size = 32
    val_batch_size = 64
    epochs = 100
    eval_per_epoch = 2
    m = nn.LogSoftmax(axis=1)
    divide=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)

    train_transforms = ds.transforms.c_transforms.Compose([
       
        ds.vision.c_transforms.RandomHorizontalFlip(),
        #ds.vision.py_transforms.ToTensor(),
        ds.vision.c_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ds.vision.c_transforms.HWC2CHW()
    ])
    val_transforms = ds.transforms.c_transforms.Compose([
       
        #ds.vision.c_transforms.RandomHorizontalFlip(),
        #ds.vision.py_transforms.ToTensor(),
        ds.vision.c_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ds.vision.c_transforms.HWC2CHW()
    ])
    ds.config.set_seed(66)
    ava_train = mydataset.GetDatasetGenerator(data_path, lable_path, divide=divide, transforms=train_transforms,ranges=0.02, gtlabel="Composition")
    train_dataset = ds.GeneratorDataset(ava_train, ["data", "label" , "second_label","length" , "target", "light", "color", "composition" ], shuffle=True) 
    train_dataset = train_dataset.batch(batch_size,drop_remainder=True)
    
    ava_val = mydataset.GetDatasetGenerator(data_path, val_path, divide=divide, transforms=val_transforms,ranges=0.02, gtlabel="Composition")
    val_dataset = ds.GeneratorDataset(ava_val, ["data",  "label" , "second_label","length" , "target", "light", "color", "composition" ], shuffle=False)
    val_dataset = val_dataset.batch(val_batch_size,drop_remainder=True)
    eval_dataset = val_dataset
    
    loss = nn.MSELoss()
    
    #param_dict = load_checkpoint("efficientnet_v111.ckpt")
    param_class = load_checkpoint("./regre2/regre2.ckpt")
    backdone = efficientnet_b0(num_classes=1000,
                               drop_rate=0.2,
                               drop_connect_rate=0.2,
                               global_pool="avg",
                               bn_tf=False,)
    load_param_into_net(backdone, param_class)
    backdone.set_train(True)
    
    net = Net(10)
    net.set_train(True)
    
    load_param_into_net(net, param_class)
    
    for para in net.trainable_params():
     
      if para.name.split('.')[0] == 'regression_composition1':
        para.requires_grad = True      
      elif para.name.split('.')[0] == 'regression_composition2':
        para.requires_grad = True
      elif para.name.split('.')[0] == 'eca5':
        para.requires_grad = True        
      else:
        para.requires_grad = False
    for para in net.trainable_params():
      print(para)         
    polynomial_decay_lr = nn.learning_rate_schedule.PolynomialDecayLR(learning_rate=0.0001,
                                                                  end_learning_rate=0.0000001,
                                                                  decay_steps=4,
                                                                  power=0.5)
    optim = nn.Adam(net.trainable_params(), learning_rate=polynomial_decay_lr,beta1=0.98 ,beta2=0.999, weight_decay=0.0001)
    #train_network = nn.SequentialCell([backdone, net])

    
    loss_net = CustomWithLossCell(backdone, net, loss)
    cb = LossMonitor()
    
    eval_net = CustomWithEvalCell(backdone, net)
    eval_net.set_train(False)
    metric = nn.MSE()
    best_class_acc = 0
    print("—————————————Start Train—————————————")
    # 获取训练过程数据
    dataset_size = train_dataset.get_dataset_size()
    model = Model(loss_net, optimizer=optim,eval_network =eval_net, metrics={"MSE": nn.MSE()})
    config_ck = CheckpointConfig(save_checkpoint_steps=eval_per_epoch*dataset_size, keep_checkpoint_max=50)
    ckpoint_cb = ModelCheckpoint(prefix="regre3",directory=ckpt_save_dir, config=config_ck)
    epoch_per_eval = {"epoch": [], "mse": []}
    eval_cb = EvalCallBack(model, eval_dataset, eval_per_epoch, epoch_per_eval)
    model.train(epoch=epochs, train_dataset=train_dataset, callbacks=[ckpoint_cb,cb,eval_cb])

