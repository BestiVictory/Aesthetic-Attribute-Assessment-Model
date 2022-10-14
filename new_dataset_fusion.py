# encoding:utf-8
import pandas as pd
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import mindspore.dataset as ds
import mindspore as ms
from mindspore import nn,Parameter,Tensor
from mindspore.ops import operations as P
import numpy as np

# encoding:utf-8
import pandas as pd
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import mindspore.dataset as ds
import mindspore as ms
from mindspore import nn,Parameter,Tensor
from mindspore.ops import operations as P
import numpy as np

def resize(img, size, interpolation=2):
    if isinstance(size, int):
        w, h = img.size
        if w >= h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        if h > w:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)
        
def pad(img, padding, fill=0, padding_mode='constant'):
    if padding_mode == 'constant':
        if img.mode == 'P':
            palette = img.getpalette()
            image = ImageOps.expand(img, border=padding, fill=fill)
            image.putpalette(palette)
            return image

        return ImageOps.expand(img, border=padding, fill=fill)

class GetDatasetGenerator:
    '''
     :rtype: pil_img, label,score, roi
    '''

    def __init__(self, root, labelroot,divide =(0.4,0.7), transforms=None, ranges=0.05, gtlabel="score"):
        self.transforms = transforms
        self.root = root
        self.label = pd.read_csv(labelroot, sep=',')
        self.filename = self.label['num'].values
        self.divide = divide
        self.ranges = ranges
        self.scoregt = self.label['score'].values      
        self.category = []
        self.gtlabel = gtlabel
        for gt in self.scoregt:
            if gt <= 0.1:
                self.category.append(0)                 
            elif gt<=0.2:
                self.category.append(1)
            elif gt<=0.3:
                self.category.append(2)
            elif gt<=0.4:
                self.category.append(3)
            elif gt<=0.5:
                self.category.append(4)
            elif gt<=0.6:
                self.category.append(5)
            elif gt<=0.7:
                self.category.append(6)
            elif gt<=0.8:
                self.category.append(7)
            elif gt<=0.9:
                self.category.append(8)
            else:
                self.category.append(9)
    
    def __getitem__(self, index):
        
        img_path = self.filename[index]

        color = self.label.loc[self.label['num'] == img_path, 'color_feature'].values
        light = self.label.loc[self.label['num'] == img_path, 'light_feature'].values
        composition = self.label.loc[self.label['num'] == img_path, 'composition_feature'].values
        
        if self.gtlabel=="score":
            gt = self.label.loc[self.label['num'] == img_path, 'score'].values
        elif self.gtlabel=="Color":
            gt = self.label.loc[self.label['num'] == img_path, 'Color'].values
        elif self.gtlabel=="Light":
            gt = self.label.loc[self.label['num'] == img_path, 'Light'].values
        else:
            gt = self.label.loc[self.label['num'] == img_path, 'Composition'].values
        
        attributes_num = len(self.divide)
        min_legth = 0.05
        second_label = 0

        if gt <= self.divide[0]:
            if min_legth > abs(self.divide[0] - gt):
                min_legth = abs(self.divide[0] - gt)
                if min_legth < self.ranges:
                    second_label = 1
                    min_legth =self.ranges
                else:
                    second_label = 0
                    min_legth = 0
            label = 0
        elif gt > self.divide[-1]:
            if min_legth > abs(self.divide[-1] - gt):
                min_legth = abs(self.divide[-1] - gt)
                if min_legth < self.ranges:
                    second_label = attributes_num - 1
                    min_legth =self.ranges
                else:
                    second_label = attributes_num
                    min_legth = 0
            label = attributes_num
        else:
            for i in range(attributes_num - 1):
                if gt > self.divide[i] and gt <= self.divide[i + 1]:
                    if min_legth > abs(self.divide[i] - gt):
                        min_legth = abs(self.divide[i] - gt)
                        if min_legth < self.ranges:
                            second_label = i
                            min_legth =self.ranges
                        else:
                            second_label = i + 1
                            min_legth = 0

                    if min_legth > abs(self.divide[i + 1] - gt):
                        min_legth = abs(self.divide[i + 1] - gt)
                        if min_legth < self.ranges:
                            second_label = i + 2
                            min_legth =self.ranges
                        else:
                            second_label = i + 1
                            min_legth = 0

                    label = i + 1

        min_legth =float(min_legth)

        pil_img1 = Image.open(self.root +  str(img_path))

        try:
            for orientation in ExifTags.TAGS.keys() : 
                if ExifTags.TAGS[orientation]=='Orientation' : 
                    break  
                exif=dict((ExifTags.TAGS[k], v) for k, v in pil_img1._getexif().items() if k in ExifTags.TAGS)  
                if exif['Orientation'] == 3: 
                    pil_img1 = pil_img1.rotate(180, expand = True)
                elif exif['Orientation'] == 6:
                    pil_img1 = pil_img1.rotate(270, expand = True)
                elif exif['Orientation'] == 8: 
                    pil_img1 = pil_img1.rotate(90, expand = True)

        except:
            pass
  
        pil_img1 = pil_img1.convert('RGB')
        pil_img1 = pil_img1.resize((224,224),Image.ANTIALIAS)
        if self.transforms:
            pil_img1 = self.transforms(pil_img1)

        light = light[0].split("[")[1].split("]")[0].split(",")
        light = np.array([float(s) for s in light])
        color = color[0].split("[")[1].split("]")[0].split(",")
        color = np.array([float(s) for s in color])
        composition = composition[0].split("[")[1].split("]")[0].split(",")
        composition = np.array([float(s) for s in composition[0:10]])

        label = np.array(label, np.int32)
        second_label = np.array(second_label, np.int32)
        gt = np.array(gt, np.float32)
        light = np.array(light, np.float32)
        color = np.array(color, np.float32)
        composition = np.array(composition, np.float32)
        
        return pil_img1, label, second_label, min_legth, gt, light, color,composition
        
        
    def __len__(self):
        return len(self.filename)
    
    def get_classes_for_all_imgs(self):
        return self.category


if __name__ == "__main__":
    data_path = "./images/"  
    lable_path = "./trainlist_c_test.csv"
    train_transforms = ds.transforms.c_transforms.Compose([
       
        ds.vision.c_transforms.RandomHorizontalFlip(),
        #ds.vision.py_transforms.ToTensor(),
        ds.vision.c_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ds.vision.c_transforms.HWC2CHW()
    ])

    ava_train = GetDatasetGenerator(data_path, lable_path, divide=0.5, transforms=train_transforms,ranges=0.05)
    train_dataset = ds.GeneratorDataset(ava_train, ["data", "label" , "second_label","length"], shuffle=False)
    train_dataset = train_dataset.batch(3)
    for data in train_dataset.create_dict_iterator():
        #print(data["data"].shape,data["label"],data["second_label"],data["length"])
        print("Done")
    
