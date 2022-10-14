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
from new_hcr_models_class import Net
from loss import My_loss
from mindspore.train.callback import LossMonitor , ModelCheckpoint, CheckpointConfig
'''
backdone = efficientnet_b0(2)
features = backdone.extract_features(output)
print(features.shape)
'''
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
        return output_score, output_light, output_color, output_composition
    
class CustomWithEvalCell(nn.Cell):
    def __init__(self, backbone,net):
        super(CustomWithEvalCell, self).__init__(auto_prefix=False)
        self._net = net
        self._backbone = backbone
        
    def construct(self, data, label, second_label ,length, target, light, color, composition):
        output = self._backbone(data)
        output_class, output_class2, output_light, output_color, output_composition, output_score = self._net(output,light, color, composition)
        return output_score, target

if __name__ == "__main__":
    
    data_path = "./images/"
    lable_path = "./datalist/train_attributes.csv"
    val_path = "./datalist/test_attributes_mix.csv"
    ckpt_save_dir = "./regre1"
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
    ava_train = mydataset.GetDatasetGenerator(data_path, lable_path, divide=divide, transforms=train_transforms,ranges=0.02)
    train_dataset = ds.GeneratorDataset(ava_train, ["data", "label" , "second_label","length" , "target", "light", "color", "composition" ], shuffle=True) 
    train_dataset = train_dataset.batch(batch_size,drop_remainder=True)
    
    ava_val = mydataset.GetDatasetGenerator(data_path, val_path, divide=divide, transforms=val_transforms,ranges=0.02)
    val_dataset = ds.GeneratorDataset(ava_val, ["data",  "label" , "second_label","length" , "target", "light", "color", "composition" ], shuffle=False)
    eval_dataset = val_dataset
    
    loss = nn.MSELoss()
    
    #param_dict = load_checkpoint("efficientnet_v111.ckpt")
    param_class = load_checkpoint("./regre4/regre4.ckpt")
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
    
        
    optim = nn.Adam(net.trainable_params(), learning_rate=0.0001,beta1=0.98 ,beta2=0.999, weight_decay=0.0001)


    
    loss_net = CustomWithLossCell(backdone, net, loss)
    cb = LossMonitor()
    
    eval_net = CustomWithEvalCell(backdone, net)
    eval_net.set_train(False)
    metric = nn.MSE()
    best_class_acc = 0
    print("—————————————Start Test—————————————")
    # 获取训练过程数据
    dataset_size = train_dataset.get_dataset_size()
    model = Model(loss_net, optimizer=optim,eval_network =eval_net, metrics={"MSE": nn.MSE()})
    config_ck = CheckpointConfig(save_checkpoint_steps=eval_per_epoch*dataset_size, keep_checkpoint_max=50)
    ckpoint_cb = ModelCheckpoint(prefix="regre1",directory=ckpt_save_dir, config=config_ck)
    epoch_per_eval = {"epoch": [], "mse": []}
    eval_cb = EvalCallBack(model, eval_dataset, eval_per_epoch, epoch_per_eval)
    #model.train(epoch=epochs, train_dataset=train_dataset, callbacks=[ckpoint_cb,cb,eval_cb])
    #eval_result = model.eval(eval_dataset)
    #print(eval_result)

    output = []
    unsqueeze= P.ExpandDims()
    label = pd.read_csv(val_path, sep=',')
    filename = label['num'].values
    
    #for img_path in filename:
        #print(img_path)
    #for para in param_dict:
      #print(param_dict[para])
    i = 0
    for data in eval_dataset.create_dict_iterator():
        img_path = filename[i]
        score = label.loc[label['num'] == img_path, 'score'].values
        color = label.loc[label['num'] == img_path, 'Color'].values
        light = label.loc[label['num'] == img_path, 'Light'].values
        composition = label.loc[label['num'] == img_path, 'Composition'].values
        
        output_score, output_light, output_color, output_composition = model.predict(unsqueeze(data["data"],0),unsqueeze(data["label"],0),unsqueeze(data["second_label"],0),unsqueeze(data["length"],0),unsqueeze(data["target"],0),unsqueeze(data["light"],0),unsqueeze(data["color"],0),unsqueeze(data["composition"],0))
        output.append([img_path, output_score[0][0], score[0], output_light[0][0], light[0], output_color[0][0], color[0], output_composition[0][0], composition[0]])
        i = i+1
        
    name=['num', 'score','gt_score','light','gt_light','color','gt_color','composition','gt_composition']
    test=pd.DataFrame(columns=name,data=output)
    test.to_csv('./output/attributes_mix.csv',encoding='utf-8',index=False)

