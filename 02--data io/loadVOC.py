import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
import cv2 as cv
import numpy as np


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, imgfile, labelfile, transform=None):
        super(VOCDataset, self).__init__()
        self.imglist = open(imgfile).read().split('\n')
        self.labelfile = labelfile
        self.transform = transform

    def __getitem__(self, index):
        img = cv.imread(self.imglist[index])
        img = img[:, :, ::-1].copy()  # RGB
        img = transform(img)
        _, name = os.path.split(self.imglist[index])
        name, _ = os.path.splitext(name)
        label = open(os.path.join(labelfile, name + '.txt')).read().split()
        label = [float(x) for x in label]
        return img, label

    def __len__(self):
        return len(self.imglist)


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
])

labelfile = '/home/wxrui/DATA/VOCdevkit/VOC2007/dst/label'

trainfile = '/home/wxrui/DATA/VOCdevkit/VOC2007/dst/2007_train.txt'
train_set = VOCDataset(imgfile=trainfile, labelfile=labelfile, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=8)

imgs, labels = train_set.__getitem__(1)
print(labels)

# 每个样本label长度不同, 使用DataLoader会丢失信息
imgs, labels = train_loader.__iter__().next()
print(labels)
