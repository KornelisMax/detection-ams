from flask import Flask
class NoneTransform(object):
    ''' Does nothing to the image. To be used instead of None '''
    
    def __call__(self, image):       
        return image

import os
import numpy as np
import torch
from PIL import Image
import math

class PennFudanDataset(object):
    def __init__(self, files, labels, y, transforms = None):
        self.root = root
        self.transforms = transforms
        self.files = files
        self.bounding_boxes = y
        self.labels = labels
        

    def __getitem__(self, idx):
        # load images ad masks
        img_path = self.files[idx]
        print(img_path)
        img = io.imread(img_path)
        # print(img)
        img1 = Image.open(img_path)

        # print(idx)
        # print(img1)
        # print("BEFORE")
        # print(img.shape)

        #if img.shape dimensions not equal 3
        
        #copy dimensions 3 times to get grayscale "mimic" , reshape it. copy from one tensor to another
        # img = Image.fromarray(np.ones((100, 100, 3), dtype=np.uint8))
        T = transforms.Compose([
            transforms.ToTensor(),            
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))  if img1.mode!='RGB'  else NoneTransform()                 
            ])    
        
        # print(T(img))
        w, h, _ = T(img).shape

        # print("AFTER")
        # print(T(img).shape)
        obj_ids = 1
        
        num_objs = 1
        
        xmin, ymin, xmax, ymax = self.bounding_boxes[idx,:4]
        boxes = []
        boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.tensor(boxes, dtype=torch.float32)

        label = self.labels[idx]
        labels = []
        labels.append(label)
        # print(labels)
        labels = torch.tensor(labels, dtype=torch.int64)
        # labels = torch.ones((1,), dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        mask = np.zeros((h, w))
        mask[xmin:xmax, ymin:ymax] = 1
        masks = mask == 1
        
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["masks"] = masks
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        target = target
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.files)
app = Flask(__name__)

@app.route('/')
def hello_world():
    
    return 'hello'
