from __future__ import print_function, absolute_import  
import os
import glob
import torch
import random
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image, ImageFile, ImageEnhance


class Generator(Dataset):
    def __init__(self, datalist, inputsize, train=False):
        self.datalist = datalist   
        self.inputsize = inputsize  
        self.train = train    
        
    def __len__(self):
        return len(self.datalist)
    
    # 原始版本
    def __getitem__(self, index):
        gt = Image.open(self.datalist[index])
        gt = gt.resize((512, 512), Image.NEAREST)
        gt = np.asarray(gt, dtype=np.float32)
        gt = torch.from_numpy(gt / 255. * 3).long() 


        img = Image.open(self.datalist[index].replace('lab', 'img')) 
        transform = transforms.Compose([transforms.Resize((self.inputsize, self.inputsize)),
                                        transforms.ToTensor()])
        
        return transform(img), gt
